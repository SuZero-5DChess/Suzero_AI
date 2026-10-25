#undef NDEBUG // make sure assert is always enabled
#include <cassert>
#include <condition_variable>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <functional>
#include <optional>
#include <utility>

#include "engine/uci.h"
#include <stop_token>
#include <chrono>

class scripted_io_handler : public io_handler
{
public:
    struct scripted_line
    {
        std::string text;
        size_t min_outputs_before_release;
    };

    explicit scripted_io_handler(std::vector<scripted_line> input_lines) : input_lines(std::move(input_lines)) {}

    std::string read_line() override
    {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [this]() {
            return next_input >= input_lines.size() || output_lines.size() >= input_lines[next_input].min_outputs_before_release;
        });

        if(next_input == input_lines.size())
            return {};
        std::string line = input_lines[next_input++].text;
        lock.unlock();
        if(input_observer)
            input_observer(line);
        return line;
    }

    void write_line(const std::string &line) override
    {
        std::lock_guard<std::mutex> lock(mutex);
        if(observer)
            observer(line);
        output_lines.push_back(line);
        cv.notify_all();
    }

    bool is_open() override
    {
        std::lock_guard<std::mutex> lock(mutex);
        return next_input < input_lines.size();
    }

    std::vector<std::string> output_lines;
    std::function<void(const std::string &)> observer;
    std::function<void(const std::string &)> input_observer;

private:
    std::vector<scripted_line> input_lines;
    size_t next_input = 0;
    std::mutex mutex;
    std::condition_variable cv;
};

class dummy_engine : public engine
{
public:
    struct option_change
    {
        std::string key;
        option_value_t value;
    };

    explicit dummy_engine(std::unique_ptr<io_handler> io) : engine(std::move(io))
    {
        is_debug_mode = false;
    }

    void initialize() override
    {
    }

    std::optional<action> find_best_move(std::optional<int>, std::optional<int>, std::stop_token stop_token) override
    {
        std::unique_lock<std::mutex> lock(search_mutex);
        while(!stop_token.stop_requested())
        {
            search_cv.wait_for(lock, std::chrono::milliseconds(10));
        }

        return std::nullopt;
    }

    bool busy_for_test() const
    {
        return is_busy();
    }

    const std::vector<option_change> &option_changes() const
    {
        return changes;
    }

protected:
    void on_option_changed(const std::string &key, const option_value_t &value) override
    {
        changes.push_back({key, value});
    }

private:
    std::mutex search_mutex;
    std::condition_variable search_cv;
    std::vector<option_change> changes;
};

class immediate_engine : public dummy_engine
{
public:
    using dummy_engine::dummy_engine;

    std::optional<action> find_best_move(
        std::optional<int>, std::optional<int>, std::stop_token) override
    {
        return std::nullopt;
    }
};

class reporting_engine : public immediate_engine
{
public:
    using immediate_engine::immediate_engine;

    std::optional<action> find_best_move(
        std::optional<int>, std::optional<int>, std::stop_token) override
    {
        send_info("depth 1");
        send_info("depth 2");
        return std::nullopt;
    }
};

class controlled_setup_engine : public immediate_engine
{
public:
    enum class blocked_operation
    {
        initialize,
        new_game
    };

    controlled_setup_engine(
        std::unique_ptr<io_handler> io,
        blocked_operation operation)
        : immediate_engine(std::move(io)), operation(operation)
    {}

    void initialize() override
    {
        if(operation == blocked_operation::initialize)
        {
            wait_for_release();
        }
    }

    void start_new_game() override
    {
        if(operation == blocked_operation::new_game)
        {
            wait_for_release();
        }
        immediate_engine::start_new_game();
    }

    void release()
    {
        std::lock_guard<std::mutex> lock(mutex);
        released = true;
        cv.notify_all();
    }

private:
    void wait_for_release()
    {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [this]() { return released; });
    }

    blocked_operation operation;
    std::mutex mutex;
    std::condition_variable cv;
    bool released = false;
};

void terminal_response_is_published_after_idle()
{
    auto io = std::make_unique<scripted_io_handler>(
        std::vector<scripted_io_handler::scripted_line>{
            {"5duci", 0},
            {"go movetime 1", 1},
            {"position startpos", 4},
            {"print", 4},
            {"quit", 5},
        });
    auto *io_ptr = io.get();
    reporting_engine eng(std::move(io));
    bool terminal_was_idle = false;
    io_ptr->observer = [&eng, &terminal_was_idle](const std::string &line) {
        if(line == "nobestmove")
        {
            terminal_was_idle = !eng.busy_for_test();
        }
    };

    eng.mainloop();

    assert(terminal_was_idle);
    const auto outputs = io_ptr->output_lines;
    assert(outputs.size() == 6);
    assert((std::vector<std::string>(outputs.begin(), outputs.begin() + 4)
        == std::vector<std::string>{
            "5duciok", "info depth 1", "info depth 2", "nobestmove"}));
    assert(outputs[4].starts_with("["));
    assert(outputs[5] == "bye");
}

void isready_is_immediate_during_search()
{
    auto io = std::make_unique<scripted_io_handler>(
        std::vector<scripted_io_handler::scripted_line>{
            {"5duci", 0},
            {"position startpos", 1},
            {"go movetime 1000", 1},
            {"isready", 1},
            {"isready", 2},
            {"position startpos moves (0T1)h2h4 submit", 3},
            {"stop", 4},
            {"quit", 5},
        });
    auto *io_ptr = io.get();
    dummy_engine eng(std::move(io));

    eng.mainloop();

    assert((io_ptr->output_lines == std::vector<std::string>{
        "5duciok",
        "readyok",
        "readyok",
        "info engine is busy, please run 'stop' first",
        "nobestmove",
        "bye",
    }));
}

void initialization_defers_every_isready()
{
    auto io = std::make_unique<scripted_io_handler>(
        std::vector<scripted_io_handler::scripted_line>{
            {"5duci", 0},
            {"isready", 0},
            {"isready", 0},
            {"stop", 0},
            {"quit", 3},
        });
    auto *io_ptr = io.get();
    controlled_setup_engine eng(
        std::move(io), controlled_setup_engine::blocked_operation::initialize);
    io_ptr->input_observer = [&eng](const std::string &line) {
        if(line == "stop")
        {
            eng.release();
        }
    };

    eng.mainloop();

    assert((io_ptr->output_lines == std::vector<std::string>{
        "5duciok", "readyok", "readyok", "bye"}));
}

void new_game_defers_every_isready()
{
    auto io = std::make_unique<scripted_io_handler>(
        std::vector<scripted_io_handler::scripted_line>{
            {"5duci", 0},
            {"5ducinewgame", 1},
            {"isready", 1},
            {"isready", 1},
            {"stop", 1},
            {"quit", 3},
        });
    auto *io_ptr = io.get();
    controlled_setup_engine eng(
        std::move(io), controlled_setup_engine::blocked_operation::new_game);
    io_ptr->input_observer = [&eng](const std::string &line) {
        if(line == "stop")
        {
            eng.release();
        }
    };

    eng.mainloop();

    assert((io_ptr->output_lines == std::vector<std::string>{
        "5duciok", "readyok", "readyok", "bye"}));
}

void quit_discards_deferred_isready()
{
    auto io = std::make_unique<scripted_io_handler>(
        std::vector<scripted_io_handler::scripted_line>{
            {"5duci", 0},
            {"isready", 0},
            {"quit", 0},
        });
    auto *io_ptr = io.get();
    controlled_setup_engine eng(
        std::move(io), controlled_setup_engine::blocked_operation::initialize);
    io_ptr->observer = [&eng](const std::string &line) {
        if(line == "bye")
        {
            eng.release();
        }
    };

    eng.mainloop();

    assert((io_ptr->output_lines == std::vector<std::string>{"bye"}));
}

void print_protocol_transcript_and_option_changes(const scripted_io_handler &io, const dummy_engine &eng)
{
    std::cout << "Transcript:\n";
    for(const auto &line : io.output_lines)
    {
        std::cout << line << '\n';
    }

    std::cout << "\non_option_changed invocations:\n";
    for(const auto &ch : eng.option_changes())
    {
        std::cout << "  " << ch.key << " -> ";
        if(std::holds_alternative<bool>(ch.value))
            std::cout << (std::get<bool>(ch.value) ? "true" : "false") << " (bool)";
        else if(std::holds_alternative<int>(ch.value))
            std::cout << std::get<int>(ch.value) << " (int)";
        else if(std::holds_alternative<double>(ch.value))
            std::cout << std::get<double>(ch.value) << " (double)";
        else if(std::holds_alternative<std::string>(ch.value))
            std::cout << '"' << std::get<std::string>(ch.value) << "\" (string)";
        else
            std::cout << "(monostate)";
        std::cout << '\n';
    }
}

void verify_option_values_and_change_notifications(const dummy_engine &eng)
{
    // Verify setoption: options stored correctly
    auto analyse = eng.get_option("UCI_AnalyseMode");
    assert(std::holds_alternative<bool>(analyse));
    assert(std::get<bool>(analyse) == true);

    auto multipv = eng.get_option("MultiPV");
    assert(std::holds_alternative<int>(multipv));
    assert(std::get<int>(multipv) == 4);

    auto strength = eng.get_option("UCI_LimitStrength");
    assert(std::holds_alternative<double>(strength));
    assert(std::get<double>(strength) == 90.5);

    auto clear = eng.get_option("Clear Hash");
    assert(std::holds_alternative<std::monostate>(clear));

    auto name = eng.get_option("UCI_EngineName");
    assert(std::holds_alternative<std::string>(name));
    assert(std::get<std::string>(name) == "MyEngine");

    // Verify on_option_changed was called for each setoption
    const auto &changes = eng.option_changes();
    assert(changes.size() == 5);
    assert(changes[0].key == "UCI_AnalyseMode");
    assert(std::holds_alternative<bool>(changes[0].value));
    assert(changes[3].key == "Clear Hash");
    assert(std::holds_alternative<std::monostate>(changes[3].value));
}

void test_scripted_protocol_and_setoption()
{
    auto io = std::make_unique<scripted_io_handler>(std::vector<scripted_io_handler::scripted_line>{
        {"5duci", 0},
        {"setoption name UCI_AnalyseMode value true", 1},
        {"setoption name MultiPV value 4", 1},
        {"setoption name UCI_LimitStrength value 90.5", 1},
        {"setoption name Clear Hash", 1},
        {"setoption name UCI_EngineName value MyEngine", 1},
        {"5ducinewgame", 1},
        {"isready", 1},
        {"position size 3x3 even fen [k*2/3/2K*:-0:1:w] [k*2/3/2K*:+0:1:w] move (0T1)c1b1 (-1T1)c1b1 submit", 2},
        {"print", 2},
        {"go depth 1 time 1", 3},
        {"isready", 3},
        {"stop", 4},
        {"isready", 5},
        {"quit", 6}
    });
    auto *io_ptr = io.get();
    dummy_engine eng(std::move(io));
    eng.mainloop();

    print_protocol_transcript_and_option_changes(*io_ptr, eng);

    // Verify basic UCI protocol
    assert(io_ptr->output_lines.size() >= 2);
    assert(io_ptr->output_lines.front() == "5duciok");
    //assert(io_ptr->output_lines[1].rfind("bestmove ", 0) == 0);

    verify_option_values_and_change_notifications(eng);
}

void test_position_defaults_to_queen_promotion()
{
    // 5DUCI positions use the protocol's default queen promotion.
    auto promotion_io = std::make_unique<scripted_io_handler>(
        std::vector<scripted_io_handler::scripted_line>{});
    dummy_engine promotion_eng(std::move(promotion_io));
    promotion_eng.set_position(
        "size 4x4 odd fen [3k/P3/4/K*3:0:1:w]",
        "(0T1)a3a4");
    assert(promotion_eng.get_current_state()->get_piece(vec4(0, 3, 1, 0), true) == QUEEN_W);
}

void test_invalid_position_history_recovers()
{
    // Invalid position histories report an error without terminating the
    // main loop. A later valid position command must still be accepted.
    auto recovery_io = std::make_unique<scripted_io_handler>(
        std::vector<scripted_io_handler::scripted_line>{
            {"position startpos moves (0T1)h2h5", 0},
            {"position startpos moves submit", 1},
            {"position startpos moves malformed", 2},
            {"position startpos moves (0T1)h2h4 submit", 3},
            {"print", 3},
            {"quit", 4}
        });
    auto *recovery_io_ptr = recovery_io.get();
    dummy_engine recovery_eng(std::move(recovery_io));
    recovery_eng.mainloop();

    assert(recovery_io_ptr->output_lines.size() == 5);
    assert(recovery_io_ptr->output_lines[0]
           == "info position error: cannot apply move (0T1)h2h5");
    assert(recovery_io_ptr->output_lines[1]
           == "info position error: cannot submit move history");
    assert(recovery_io_ptr->output_lines[2].starts_with(
        "info position error: cannot parse move malformed:"));
    assert(recovery_io_ptr->output_lines[3].find("7P") != std::string::npos);
    assert(recovery_io_ptr->output_lines[4] == "bye");
}

void test_invalid_source_and_square_rejection()
{
    auto io = std::make_unique<scripted_io_handler>(
        std::vector<scripted_io_handler::scripted_line>{});
    auto *io_ptr = io.get();
    dummy_engine eng(std::move(io));
    const std::string position = "size 4x4 odd fen [3k/P3/4/K*3:0:1:w]";
    eng.set_position(position, "");
    const auto before = eng.get_current_state()->get_boards();
    const auto present = eng.get_current_state()->get_present();
    for(const std::string &moves : {
        "(999T1)a3a4", "(999T1)a3a2", "(0T0)a3a4",
        "(0T1)e3a4", "(0T1)a3e4", "(0T1)b2b3", "(0T1)d4c4",
        "(0T1)a3a4 (0T1)a1b1"})
    {
        const auto outputs = io_ptr->output_lines.size();
        eng.set_position(position, moves);
        assert(io_ptr->output_lines.size() == outputs + 1);
        assert(io_ptr->output_lines.back().starts_with(
            "info position error: cannot apply move "));
        assert(eng.get_current_state()->get_boards() == before);
        assert(eng.get_current_state()->get_present() == present);
    }
    eng.set_position(position, "(0T1)a3a4");
    assert(eng.get_current_state()->get_piece(vec4(0, 3, 1, 0), true) == QUEEN_W);
}

void test_position_royal_safety()
{
    auto io = std::make_unique<scripted_io_handler>(
        std::vector<scripted_io_handler::scripted_line>{});
    auto *output = io.get();
    dummy_engine eng(std::move(io));
    eng.set_position("startpos", "");
    const auto before = eng.get_current_state()->get_boards();
    eng.set_position("size 8x8 odd fen [R6k/8/8/8/8/8/8/K7:0:1:w]", "");
    assert(output->output_lines.size() == 1);
    assert(output->output_lines.back() == "info position error: initial setup exposes a royal");
    assert(eng.get_current_state()->get_boards() == before);
    const std::string position = "size 8x8 odd fen [r6k/8/8/8/8/8/R7/K7:0:1:w]";
    eng.set_position(position, "(0T1)a2b2 submit");
    assert(output->output_lines.size() == 2);
    assert(output->output_lines.back() == "info position error: cannot submit move history");
    assert(eng.get_current_state()->get_boards() == before);
    // An incomplete draft may expose a royal until the action is submitted.
    eng.set_position(position, "(0T1)a2b2");
    assert(eng.get_current_state()->test_present());
    assert(!eng.get_current_state()->can_submit());
}

void test_debug_output_identifies_option_types()
{
    // Debug output uses readable type names for every option alternative.
    auto debug_io = std::make_unique<scripted_io_handler>(
        std::vector<scripted_io_handler::scripted_line>{});
    auto *debug_io_ptr = debug_io.get();
    dummy_engine debug_eng(std::move(debug_io));
    debug_eng.set_option("debug", true);
    debug_eng.set_option("button", std::monostate{});
    debug_eng.set_option("count", 4);
    debug_eng.set_option("weight", 0.5);
    debug_eng.set_option("name", std::string("MyEngine"));
    assert((debug_io_ptr->output_lines == std::vector<std::string>{
        "info [debug] option changed: debug type bool",
        "info [debug] option changed: button type monostate",
        "info [debug] option changed: count type int",
        "info [debug] option changed: weight type double",
        "info [debug] option changed: name type string"}));
}

int main()
{
    std::cerr << "Running scripted protocol tests\n";
    test_scripted_protocol_and_setoption();
    test_position_defaults_to_queen_promotion();
    test_invalid_position_history_recovers();
    test_invalid_source_and_square_rejection();
    test_position_royal_safety();
    test_debug_output_identifies_option_types();
    std::cout << "All scripted protocol tests passed.\n";

    std::cerr << "Running task completion tests\n";
    terminal_response_is_published_after_idle();
    isready_is_immediate_during_search();
    initialization_defers_every_isready();
    new_game_defers_every_isready();
    quit_discards_deferred_isready();
    std::cout << "All task completion tests passed.\n";
    return 0;
}
