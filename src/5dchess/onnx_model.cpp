// SPDX-License-Identifier: MIT
// Copyright (c) 2026 SuZero-5DChess contributors

#include "onnx_model.h"

#include "onnx_batch_profile.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace suzero::chess5d::internal {

namespace {

// ------------------------------------------------------------- metadata --

void skipWhitespace(const std::string& text, std::size_t& pos) {
  while (pos < text.size()) {
    const char ch = text[pos];
    if (ch != ' ' && ch != '\t' && ch != '\r' && ch != '\n') {
      break;
    }
    ++pos;
  }
}

void expectChar(const std::string& text, std::size_t& pos, char expected) {
  skipWhitespace(text, pos);
  if (pos >= text.size() || text[pos] != expected) {
    throw std::runtime_error("Malformed ONNX metadata JSON.");
  }
  ++pos;
}

std::string parseString(const std::string& text, std::size_t& pos) {
  expectChar(text, pos, '"');
  std::string value;
  while (pos < text.size()) {
    const char ch = text[pos++];
    if (ch == '"') {
      return value;
    }
    if (static_cast<unsigned char>(ch) < 0x20) {
      throw std::runtime_error("Malformed ONNX metadata JSON string.");
    }
    if (ch != '\\') {
      value.push_back(ch);
      continue;
    }
    if (pos >= text.size()) {
      throw std::runtime_error("Malformed ONNX metadata JSON escape.");
    }
    const char escaped = text[pos++];
    switch (escaped) {
      case '"':
      case '\\':
      case '/':
        value.push_back(escaped);
        break;
      case 'b':
        value.push_back('\b');
        break;
      case 'f':
        value.push_back('\f');
        break;
      case 'n':
        value.push_back('\n');
        break;
      case 'r':
        value.push_back('\r');
        break;
      case 't':
        value.push_back('\t');
        break;
      case 'u': {
        // The protocol fields are ASCII. Preserve the escape verbatim so a
        // decoded key cannot impersonate a protocol field.
        std::string raw = "\\u";
        for (int i = 0; i < 4; ++i) {
          if (pos >= text.size()) {
            throw std::runtime_error("Malformed ONNX metadata unicode escape.");
          }
          raw.push_back(text[pos++]);
        }
        value += raw;
        break;
      }
      default:
        throw std::runtime_error("Malformed ONNX metadata JSON escape.");
    }
  }
  throw std::runtime_error("Unterminated ONNX metadata JSON string.");
}

void skipValue(const std::string& text, std::size_t& pos, int depth);

void skipContainer(const std::string& text, std::size_t& pos, int depth) {
  const char open = text[pos];
  const char close = open == '{' ? '}' : ']';
  const bool object = open == '{';
  ++pos;
  skipWhitespace(text, pos);
  if (pos < text.size() && text[pos] == close) {
    ++pos;
    return;
  }
  while (true) {
    if (object) {
      (void)parseString(text, pos);
      expectChar(text, pos, ':');
    }
    skipValue(text, pos, depth + 1);
    skipWhitespace(text, pos);
    if (pos >= text.size()) {
      throw std::runtime_error("Unterminated ONNX metadata JSON container.");
    }
    if (text[pos] == ',') {
      ++pos;
      continue;
    }
    if (text[pos] == close) {
      ++pos;
      return;
    }
    throw std::runtime_error("Malformed ONNX metadata JSON container.");
  }
}

void skipValue(const std::string& text, std::size_t& pos, int depth) {
  if (depth > 64) {
    throw std::runtime_error("ONNX metadata JSON nesting is too deep.");
  }
  skipWhitespace(text, pos);
  if (pos >= text.size()) {
    throw std::runtime_error("Truncated ONNX metadata JSON.");
  }
  if (text[pos] == '"') {
    (void)parseString(text, pos);
    return;
  }
  if (text[pos] == '{' || text[pos] == '[') {
    skipContainer(text, pos, depth);
    return;
  }
  const std::size_t start = pos;
  while (pos < text.size()) {
    const char ch = text[pos];
    if (ch == ',' || ch == '}' || ch == ']' || ch == ' ' || ch == '\t' ||
        ch == '\r' || ch == '\n') {
      break;
    }
    ++pos;
  }
  if (pos == start) {
    throw std::runtime_error("Malformed ONNX metadata JSON value.");
  }
}

using MetadataFields = std::unordered_map<std::string, std::string>;

/// Parses the exporter's flat sidecar, keeping top-level scalar values (as
/// unescaped strings or raw number tokens) and skipping nested containers.
MetadataFields parseMetadata(const std::string& text) {
  MetadataFields fields;
  std::size_t pos = 0;
  expectChar(text, pos, '{');
  skipWhitespace(text, pos);
  if (pos < text.size() && text[pos] == '}') {
    ++pos;
  } else {
    while (true) {
      const std::string key = parseString(text, pos);
      expectChar(text, pos, ':');
      skipWhitespace(text, pos);
      if (pos >= text.size()) {
        throw std::runtime_error("Truncated ONNX metadata JSON.");
      }
      if (text[pos] == '"') {
        fields[key] = parseString(text, pos);
      } else if (text[pos] == '{' || text[pos] == '[') {
        skipContainer(text, pos, 0);
      } else {
        const std::size_t start = pos;
        skipValue(text, pos, 0);
        fields[key] = text.substr(start, pos - start);
      }
      skipWhitespace(text, pos);
      if (pos >= text.size()) {
        throw std::runtime_error("Truncated ONNX metadata JSON.");
      }
      if (text[pos] == ',') {
        ++pos;
        continue;
      }
      if (text[pos] == '}') {
        ++pos;
        break;
      }
      throw std::runtime_error("Malformed ONNX metadata JSON object.");
    }
  }
  skipWhitespace(text, pos);
  if (pos != text.size()) {
    throw std::runtime_error("Trailing data after ONNX metadata JSON.");
  }
  return fields;
}

std::string readTextFile(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("Cannot open ONNX metadata: " + path);
  }
  std::ostringstream buffer;
  buffer << input.rdbuf();
  return buffer.str();
}

void requireField(const MetadataFields& fields, const std::string& key,
                  const std::string& expected) {
  const auto found = fields.find(key);
  if (found == fields.end() || found->second != expected) {
    throw std::runtime_error("Incompatible ONNX metadata: " + key +
                             " must be \"" + expected + "\".");
  }
}

void validateMetadata(const OnnxEvaluatorConfig& config) {
  const std::string path =
      config.metadata_path.empty() ? config.model_path + ".json"
                                   : config.metadata_path;
  const MetadataFields fields = parseMetadata(readTextFile(path));
  requireField(fields, "io_protocol", "smallstep_movekey_v1");
  requireField(fields, "action_encoding", "smallstep_v1");
  requireField(fields, "action_input_encoding", "movekey_int64_v1");

  const auto checkInteger = [&](const char* key, int expected) {
    const auto found = fields.find(key);
    if (found == fields.end()) {
      return;
    }
    int parsed = 0;
    try {
      std::size_t used = 0;
      parsed = std::stoi(found->second, &used);
      if (used != found->second.size()) {
        throw std::runtime_error("trailing characters");
      }
    } catch (const std::exception&) {
      throw std::runtime_error("Malformed ONNX metadata integer: " +
                               std::string(key));
    }
    if (parsed != expected) {
      throw std::runtime_error("ONNX metadata " + std::string(key) +
                               " differs from the configured geometry.");
    }
  };
  checkInteger("board_side", config.board_side);
  checkInteger("board_squares", config.board_squares);
  checkInteger("piece_channels", config.piece_channels);
}

// -------------------------------------------------------------- tensors --

bool isFloatType(ONNXTensorElementDataType type) {
  return type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
         type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
}

float readScalar(const Ort::Value& value, std::size_t index) {
  const auto info = value.GetTensorTypeAndShapeInfo();
  const auto type = info.GetElementType();
  float result = 0.0f;
  if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    result = value.GetTensorData<float>()[index];
  } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
    result = value.GetTensorData<Ort::Float16_t>()[index].ToFloat();
  } else {
    throw std::runtime_error("Unsupported ONNX output element type.");
  }
  if (!std::isfinite(result)) {
    throw std::runtime_error("Non-finite ONNX output.");
  }
  return result;
}

void expectShape(const Ort::Value& value, const std::vector<int64_t>& expected) {
  const auto shape = value.GetTensorTypeAndShapeInfo().GetShape();
  if (shape != expected) {
    throw std::runtime_error("smallstep_movekey_v1 ONNX output shape mismatch.");
  }
}

/// Clamps logits and normalizes them with the same guarded softmax as the
/// 5dc_ml runner.
mcts::Evaluation makeEvaluation(const std::vector<float>& raw_logits,
                                float value) {
  if (raw_logits.empty()) {
    throw std::runtime_error("ONNX policy has no entries.");
  }

  std::vector<float> logits = raw_logits;
  float maximum = -std::numeric_limits<float>::infinity();
  for (float& logit : logits) {
    logit = std::clamp(logit, -20.0f, 20.0f);
    maximum = std::max(maximum, logit);
  }
  float total = 0.0f;
  for (float& logit : logits) {
    logit = std::exp(logit - maximum);
    total += logit;
  }
  if (!(total > 0.0f) || !std::isfinite(total)) {
    throw std::runtime_error("Degenerate ONNX policy.");
  }

  mcts::Evaluation evaluation;
  evaluation.policy.resize(logits.size());
  for (std::size_t i = 0; i < logits.size(); ++i) {
    evaluation.policy[i] = logits[i] / total;
  }
  evaluation.utility = static_cast<double>(value);
  return evaluation;
}

struct ActionLayout {
  std::size_t q = 0;
  bool has_submit = false;
};

ActionLayout analyzeActions(const std::vector<Action>& actions) {
  ActionLayout layout;
  for (std::size_t i = 0; i < actions.size(); ++i) {
    const Action& action = actions[i];
    if (action.is_submit) {
      if (layout.has_submit || i + 1 != actions.size()) {
        throw std::logic_error(
            "smallstep_movekey_v1 requires a single SUBMIT after all move "
            "actions");
      }
      layout.has_submit = true;
    } else {
      ++layout.q;
    }
  }
  return layout;
}

void validateEncoded(const EncodedPosition& encoded,
                     const OnnxEvaluatorConfig& config) {
  const std::size_t n = static_cast<std::size_t>(encoded.num_boards);
  const std::size_t c = static_cast<std::size_t>(encoded.piece_channels);
  const std::size_t s = static_cast<std::size_t>(encoded.board_squares);
  if (n == 0) {
    throw std::runtime_error(
        "smallstep_movekey_v1 inference requires at least one board");
  }
  if (encoded.board_planes.size() != n * c * s ||
      encoded.l_coords.size() != n || encoded.t_coords.size() != n ||
      static_cast<std::size_t>(encoded.piece_channels) !=
          static_cast<std::size_t>(config.piece_channels) ||
      static_cast<std::size_t>(encoded.board_squares) !=
          static_cast<std::size_t>(config.board_squares)) {
    throw std::runtime_error("Invalid encoded board dimensions.");
  }
}

}  // namespace

struct OnnxModel::Impl {
  explicit Impl(OnnxEvaluatorConfig cfg)
      : env(ORT_LOGGING_LEVEL_ERROR, "suzero_5dchess"),
        config(std::move(cfg)),
        memory(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,
                                          OrtMemTypeCPUInput)) {
    if (config.model_path.empty()) {
      throw std::invalid_argument("OnnxEvaluator: model_path is required");
    }
    if (config.provider != "cpu" && config.provider != "cuda") {
      throw std::invalid_argument(
          "OnnxEvaluator: provider must be \"cpu\" or \"cuda\"");
    }
    if (config.board_side <= 0 || config.board_squares <= 0 ||
        config.piece_channels <= 0 ||
        config.board_squares != config.board_side * config.board_side) {
      throw std::invalid_argument("OnnxEvaluator: invalid board geometry");
    }

    validateMetadata(config);

    if (config.provider == "cuda") {
      // AppendExecutionProvider_CUDA fails when the provider library or its
      // CUDA/cuDNN dependencies are missing, so a CUDA request never degrades
      // into a silent CPU run.
      const std::vector<std::string> providers = Ort::GetAvailableProviders();
      bool cuda_available = false;
      for (const std::string& name : providers) {
        if (name == "CUDAExecutionProvider") {
          cuda_available = true;
          break;
        }
      }
      if (!cuda_available) {
        throw std::runtime_error(
            "OnnxEvaluator: CUDAExecutionProvider is not available in this "
            "ONNX Runtime build. Install onnxruntime-gpu and make sure the "
            "CUDA/cuDNN DLLs are on PATH.");
      }
      if (config.cuda_device_id < 0) {
        throw std::invalid_argument(
            "OnnxEvaluator: cuda_device_id must be non-negative");
      }
      OrtCUDAProviderOptions cuda_options{};
      cuda_options.device_id = config.cuda_device_id;
      cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
      options.AppendExecutionProvider_CUDA(cuda_options);
    }

    options.SetIntraOpNumThreads(std::max(1, config.intra_op_threads));
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    const std::filesystem::path model_path(config.model_path);
    session = std::make_unique<Ort::Session>(env, model_path.c_str(), options);
    validateSession();
    profile.arm();
  }

  void validateSession() {
    if (session->GetInputCount() != kInputNames.size() ||
        session->GetOutputCount() != kOutputNames.size()) {
      throw std::runtime_error(
          "smallstep_movekey_v1 requires exactly 7 ONNX inputs and 3 outputs.");
    }

    Ort::AllocatorWithDefaultOptions allocator;
    constexpr std::array<std::size_t, 7> kRanks{4, 2, 2, 1, 1, 1, 3};
    for (std::size_t i = 0; i < kInputNames.size(); ++i) {
      const auto name = session->GetInputNameAllocated(i, allocator);
      if (std::string(name.get()) != kInputNames[i]) {
        throw std::runtime_error(
            "smallstep_movekey_v1 ONNX input name/order mismatch.");
      }
      const auto type_info = session->GetInputTypeInfo(i);
      const auto tensor = type_info.GetTensorTypeAndShapeInfo();
      const auto shape = tensor.GetShape();
      const auto type = tensor.GetElementType();
      input_types[i] = type;
      const bool floating_input = i == 0 || i == 4;
      if (shape.size() != kRanks[i] ||
          (floating_input ? !isFloatType(type)
                          : type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)) {
        throw std::runtime_error(
            "smallstep_movekey_v1 ONNX input rank/type mismatch.");
      }
      if ((i == 0 && (shape[2] != config.piece_channels ||
                      shape[3] != config.board_squares)) ||
          (i == 6 && shape[2] != kMoveKeyFields)) {
        throw std::runtime_error(
            "ONNX board geometry differs from the configured variant.");
      }
      const std::size_t dynamic_dims = (i == 0) ? 2 : (i == 6 ? 2 : kRanks[i]);
      for (std::size_t axis = 0; axis < dynamic_dims && axis < shape.size();
           ++axis) {
        if (shape[axis] >= 0) {
          throw std::runtime_error(
              "smallstep_movekey_v1 ONNX requires dynamic B/N/q axes.");
        }
      }
    }

    for (std::size_t i = 0; i < kOutputNames.size(); ++i) {
      const auto name = session->GetOutputNameAllocated(i, allocator);
      if (std::string(name.get()) != kOutputNames[i]) {
        throw std::runtime_error(
            "smallstep_movekey_v1 ONNX output name/order mismatch.");
      }
      const auto type_info = session->GetOutputTypeInfo(i);
      const auto tensor = type_info.GetTensorTypeAndShapeInfo();
      const auto shape = tensor.GetShape();
      if (!isFloatType(tensor.GetElementType()) || shape.size() != 2 ||
          shape[0] >= 0 || (i < 2 ? shape[1] != 1 : shape[1] >= 0)) {
        throw std::runtime_error(
            "smallstep_movekey_v1 ONNX output signature mismatch.");
      }
    }
  }

  /// Builds a float tensor, converting to fp16 when the model expects it.
  /// `half` must outlive the returned Ort::Value.
  Ort::Value makeFloatTensor(std::size_t input, std::vector<float>& data,
                             std::size_t count,
                             const std::vector<int64_t>& shape,
                             std::vector<Ort::Float16_t>& half) {
    if (input_types[input] == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      return Ort::Value::CreateTensor<float>(memory, data.data(), count,
                                             shape.data(), shape.size());
    }
    half.clear();
    half.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
      half.emplace_back(data[i]);
    }
    return Ort::Value::CreateTensor<Ort::Float16_t>(
        memory, half.data(), half.size(), shape.data(), shape.size());
  }

  Ort::Value makeIntTensor(std::vector<int64_t>& data,
                           const std::vector<int64_t>& shape) {
    return Ort::Value::CreateTensor<int64_t>(memory, data.data(), data.size(),
                                             shape.data(), shape.size());
  }

  // Release providers before DLL detach, after all session-owned resources.
  Ort::Env env;
  OnnxBatchProfile profile;
  OnnxEvaluatorConfig config;
  Ort::SessionOptions options;
  Ort::MemoryInfo memory;
  std::unique_ptr<Ort::Session> session;
  std::array<ONNXTensorElementDataType, 7> input_types{};
  std::mutex mutex;

  static constexpr std::array<const char*, 7> kInputNames{
      "board_planes", "l_coords",  "t_coords",    "used_board_counts",
      "urgency",      "side_to_move", "actions"};
  static constexpr std::array<const char*, 3> kOutputNames{
      "value", "submit_logit", "other_logits"};
};

OnnxModel::OnnxModel(OnnxEvaluatorConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

OnnxModel::~OnnxModel() = default;

const OnnxEvaluatorConfig& OnnxModel::config() const { return impl_->config; }

mcts::Evaluation OnnxModel::evaluate(const IChess5DModelInput& input) {
  const auto& actions = input.actions();
  const ActionLayout layout = analyzeActions(actions);

  // The protocol requires q > 0 for every inference request. A position whose
  // only candidate is SUBMIT is forced and needs no network.
  if (layout.q == 0) {
    if (!layout.has_submit) {
      throw std::runtime_error(
          "OnnxEvaluator: terminal states have no policy to evaluate");
    }
    mcts::Evaluation forced;
    forced.policy.assign(1, 1.0f);
    forced.utility = 0.0;
    return forced;
  }

  EncodedPosition encoded = input.encodePosition();
  validateEncoded(encoded, impl_->config);

  const std::size_t n = static_cast<std::size_t>(encoded.num_boards);
  const std::size_t c = static_cast<std::size_t>(encoded.piece_channels);
  const std::size_t s = static_cast<std::size_t>(encoded.board_squares);
  const int64_t board_count = static_cast<int64_t>(n);
  const int64_t q_count = static_cast<int64_t>(layout.q);
  const std::vector<int64_t> board_shape{1, board_count,
                                         static_cast<int64_t>(c),
                                         static_cast<int64_t>(s)};
  const std::vector<int64_t> coord_shape{1, board_count};
  const std::vector<int64_t> scalar_shape{1};
  const std::vector<int64_t> action_shape{1, q_count, kMoveKeyFields};

  std::vector<int64_t> action_data(layout.q * kMoveKeyFields, 0);
  for (std::size_t i = 0; i < layout.q; ++i) {
    const MoveKey& key = actions[i].move;
    std::copy(key.begin(), key.end(),
              action_data.begin() +
                  static_cast<std::ptrdiff_t>(i * kMoveKeyFields));
  }
  std::vector<int64_t> used_board_counts{board_count};
  std::vector<int64_t> side_to_move{static_cast<int64_t>(encoded.side_to_move)};
  std::vector<float> urgency{encoded.urgency};

  // Float16 models get a one-shot conversion buffer. Both buffers stay alive
  // until the synchronous Run() returns.
  std::vector<Ort::Float16_t> board_half;
  std::vector<Ort::Float16_t> urgency_half;

  std::array<Ort::Value, 7> inputs = {
      impl_->makeFloatTensor(0, encoded.board_planes, n * c * s, board_shape,
                             board_half),
      impl_->makeIntTensor(encoded.l_coords, coord_shape),
      impl_->makeIntTensor(encoded.t_coords, coord_shape),
      impl_->makeIntTensor(used_board_counts, scalar_shape),
      impl_->makeFloatTensor(4, urgency, 1, scalar_shape, urgency_half),
      impl_->makeIntTensor(side_to_move, scalar_shape),
      impl_->makeIntTensor(action_data, action_shape)};

  std::vector<Ort::Value> outputs;
  {
    const std::lock_guard<std::mutex> lock(impl_->mutex);
    outputs = impl_->session->Run(
        Ort::RunOptions{nullptr}, Impl::kInputNames.data(), inputs.data(),
        inputs.size(), Impl::kOutputNames.data(), Impl::kOutputNames.size());
  }
  if (outputs.size() != 3) {
    throw std::runtime_error("Unexpected ONNX output count.");
  }
  expectShape(outputs[0], {1, 1});
  expectShape(outputs[1], {1, 1});
  expectShape(outputs[2], {1, q_count});

  std::vector<float> logits;
  logits.reserve(layout.q + (layout.has_submit ? 1 : 0));
  for (std::size_t i = 0; i < layout.q; ++i) {
    logits.push_back(readScalar(outputs[2], i));
  }
  if (layout.has_submit) {
    logits.push_back(readScalar(outputs[1], 0));
  }
  if (logits.size() != actions.size()) {
    throw std::runtime_error("ONNX policy size does not match candidate actions.");
  }
  return makeEvaluation(logits, readScalar(outputs[0], 0));
}

std::vector<mcts::Evaluation> OnnxModel::evaluateBatch(
    const std::vector<const IChess5DModelInput*>& model_inputs) {
  OnnxBatchProfile::Call profile_call;
  profile_call.started = impl_->profile.stamp();
  std::vector<mcts::Evaluation> results(model_inputs.size());

  struct Pending {
    std::size_t result_index = 0;
    EncodedPosition encoded;
    std::size_t q = 0;
    bool has_submit = false;
  };
  std::vector<Pending> pending;
  pending.reserve(model_inputs.size());

  profile_call.encode_begin = impl_->profile.stamp();
  for (std::size_t i = 0; i < model_inputs.size(); ++i) {
    if (model_inputs[i] == nullptr) {
      throw std::invalid_argument("OnnxModel: null model input");
    }
    const IChess5DModelInput& input = *model_inputs[i];
    const ActionLayout layout = analyzeActions(input.actions());
    if (layout.q == 0) {
      if (!layout.has_submit) {
        throw std::runtime_error(
            "OnnxEvaluator: terminal states have no policy to evaluate");
      }
      results[i].policy.assign(1, 1.0f);
      results[i].utility = 0.0;
      continue;
    }

    Pending item;
    item.result_index = i;
    item.encoded = input.encodePosition();
    item.q = layout.q;
    item.has_submit = layout.has_submit;
    validateEncoded(item.encoded, impl_->config);
    pending.push_back(std::move(item));
  }
  profile_call.encode_end = impl_->profile.stamp();

  if (pending.empty()) {
    profile_call.backend_rows =
        static_cast<std::uint64_t>(model_inputs.size());
    profile_call.forced_rows =
        static_cast<std::uint64_t>(model_inputs.size());
    profile_call.finished = impl_->profile.stamp();
    impl_->profile.record(profile_call);
    return results;
  }

  std::size_t max_boards = 0;
  std::size_t max_q = 0;
  for (const Pending& item : pending) {
    max_boards = std::max(max_boards,
                          static_cast<std::size_t>(item.encoded.num_boards));
    max_q = std::max(max_q, item.q);
  }

  const std::size_t batch = pending.size();
  const std::size_t channels =
      static_cast<std::size_t>(impl_->config.piece_channels);
  const std::size_t squares =
      static_cast<std::size_t>(impl_->config.board_squares);
  const std::size_t boards_per_row = max_boards * channels * squares;

  profile_call.allocate_begin = impl_->profile.stamp();
  std::vector<float> board_planes(batch * boards_per_row, 0.0f);
  std::vector<int64_t> l_coords(batch * max_boards, 0);
  std::vector<int64_t> t_coords(batch * max_boards, 0);
  std::vector<int64_t> used_board_counts(batch, 0);
  std::vector<float> urgency(batch, 0.0f);
  std::vector<int64_t> side_to_move(batch, 0);
  std::vector<int64_t> action_data(batch * max_q * kMoveKeyFields, 0);
  profile_call.allocate_end = impl_->profile.stamp();

  profile_call.pack_begin = impl_->profile.stamp();
  for (std::size_t b = 0; b < batch; ++b) {
    const Pending& item = pending[b];
    const std::size_t n = static_cast<std::size_t>(item.encoded.num_boards);
    const std::size_t plane_offset = b * boards_per_row;
    std::copy(item.encoded.board_planes.begin(),
              item.encoded.board_planes.end(),
              board_planes.begin() +
                  static_cast<std::ptrdiff_t>(plane_offset));
    std::copy(item.encoded.l_coords.begin(), item.encoded.l_coords.end(),
              l_coords.begin() + static_cast<std::ptrdiff_t>(b * max_boards));
    std::copy(item.encoded.t_coords.begin(), item.encoded.t_coords.end(),
              t_coords.begin() + static_cast<std::ptrdiff_t>(b * max_boards));
    used_board_counts[b] = static_cast<int64_t>(n);
    urgency[b] = item.encoded.urgency;
    side_to_move[b] = static_cast<int64_t>(item.encoded.side_to_move);

    const auto& actions = model_inputs[item.result_index]->actions();
    for (std::size_t j = 0; j < item.q; ++j) {
      const MoveKey& key = actions[j].move;
      const std::size_t offset = (b * max_q + j) * kMoveKeyFields;
      std::copy(key.begin(), key.end(),
                action_data.begin() + static_cast<std::ptrdiff_t>(offset));
    }
  }
  profile_call.pack_end = impl_->profile.stamp();

  profile_call.backend_rows =
      static_cast<std::uint64_t>(model_inputs.size());
  profile_call.inference_rows = static_cast<std::uint64_t>(pending.size());
  profile_call.forced_rows = static_cast<std::uint64_t>(
      model_inputs.size() - pending.size());
  profile_call.max_boards = static_cast<std::uint64_t>(max_boards);
  profile_call.max_q = static_cast<std::uint64_t>(max_q);
  profile_call.board_slots = static_cast<std::uint64_t>(batch) *
                             static_cast<std::uint64_t>(max_boards);
  profile_call.action_slots = static_cast<std::uint64_t>(batch) *
                              static_cast<std::uint64_t>(max_q);
  if (impl_->profile.enabled()) {
    for (const Pending& item : pending) {
      profile_call.used_boards +=
          static_cast<std::uint64_t>(item.encoded.num_boards);
      profile_call.used_actions += static_cast<std::uint64_t>(item.q);
    }
  }

  const int64_t batch_count = static_cast<int64_t>(batch);
  const int64_t board_count = static_cast<int64_t>(max_boards);
  const int64_t q_count = static_cast<int64_t>(max_q);
  const std::vector<int64_t> board_shape{
      batch_count, board_count, static_cast<int64_t>(channels),
      static_cast<int64_t>(squares)};
  const std::vector<int64_t> coord_shape{batch_count, board_count};
  const std::vector<int64_t> scalar_shape{batch_count};
  const std::vector<int64_t> action_shape{batch_count, q_count,
                                          kMoveKeyFields};

  std::vector<Ort::Float16_t> board_half;
  std::vector<Ort::Float16_t> urgency_half;

  std::array<Ort::Value, 7> inputs = {
      impl_->makeFloatTensor(0, board_planes, board_planes.size(), board_shape,
                             board_half),
      impl_->makeIntTensor(l_coords, coord_shape),
      impl_->makeIntTensor(t_coords, coord_shape),
      impl_->makeIntTensor(used_board_counts, scalar_shape),
      impl_->makeFloatTensor(4, urgency, urgency.size(), scalar_shape,
                             urgency_half),
      impl_->makeIntTensor(side_to_move, scalar_shape),
      impl_->makeIntTensor(action_data, action_shape)};

  std::vector<Ort::Value> outputs;
  // prepare_ns runs from `started` to here; encode/allocate/pack above are
  // nested subsets. run_ns covers only Session::Run, including ORT's
  // synchronous transfers, and is not pure GPU/kernel time. decode_ns covers
  // output validation, scalar reads and softmax below.
  profile_call.lock_begin = impl_->profile.stamp();
  {
    std::unique_lock<std::mutex> lock(impl_->mutex);
    profile_call.run_begin = impl_->profile.stamp();
    outputs = impl_->session->Run(
        Ort::RunOptions{nullptr}, Impl::kInputNames.data(), inputs.data(),
        inputs.size(), Impl::kOutputNames.data(), Impl::kOutputNames.size());
    profile_call.run_end = impl_->profile.stamp();
  }
  if (outputs.size() != 3) {
    throw std::runtime_error("Unexpected ONNX output count.");
  }
  expectShape(outputs[0], {batch_count, 1});
  expectShape(outputs[1], {batch_count, 1});
  expectShape(outputs[2], {batch_count, q_count});

  for (std::size_t b = 0; b < batch; ++b) {
    const Pending& item = pending[b];
    std::vector<float> logits;
    logits.reserve(item.q + (item.has_submit ? 1 : 0));
    const std::size_t base = b * max_q;
    for (std::size_t j = 0; j < item.q; ++j) {
      logits.push_back(readScalar(outputs[2], base + j));
    }
    if (item.has_submit) {
      logits.push_back(readScalar(outputs[1], b));
    }
    if (logits.size() !=
        model_inputs[item.result_index]->actions().size()) {
      throw std::runtime_error(
          "ONNX policy size does not match candidate actions.");
    }
    results[item.result_index] =
        makeEvaluation(logits, readScalar(outputs[0], b));
  }

  // Commit only after encode, Run and decode all succeeded. A forced-only
  // call never reaches this point with ran == true and never counts an ORT run.
  profile_call.finished = impl_->profile.stamp();
  profile_call.ran = true;
  impl_->profile.record(profile_call);
  return results;
}

}  // namespace suzero::chess5d::internal
