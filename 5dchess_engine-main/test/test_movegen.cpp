#undef NDEBUG
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <set>
#include <string>
#include <tuple>
#include <vector>
#include "core/check_position.h"
#include "core/multiverse.h"
#include "core/hypercuboid.h"

namespace {
using square = std::tuple<int, int, int, int>; // x, y, t, l
using moves = std::set<square>;
using squares = std::array<piece_t, BOARD_SIZE>;
int failures = 0;
bool logging = false;
std::string test_group;

void log_case(const std::string &name, const std::vector<boards_info_t> &boards,
              size_t expected_count)
{
    if (!logging) return;
    std::cout << "\nTest group: " << test_group << "\nCase: " << name << "\nFEN:\n";
    for (const auto &[l,t,c,fen] : boards)
        std::cout << '[' << fen << ':' << l << ':' << t << ':'
                  << (c ? 'b' : 'w') << "]\n";
    std::cout << "Expected moves: " << expected_count << '\n';
}

square at(vec4 p) { return {p.x(), p.y(), p.t(), p.l()}; }

// Preserve explicit unmoved markers; board::set_piece clears that flag.
std::string fen_of(const squares &b, int width = 8, int height = 8)
{
    std::string fen;
    for (int y = height - 1; y >= 0; --y) {
        int blanks = 0;
        for (int x = 0; x < width; ++x) {
            const piece_t piece = b[ppos(x,y)];
            if (piece == NO_PIECE) { ++blanks; continue; }
            if (blanks) { fen += char('0'+blanks); blanks = 0; }
            fen += char(piece_name(piece));
            if (piece_umove_flag(piece)) fen += '*';
        }
        if (blanks) fen += char('0'+blanks);
        if (y) fen += '/';
    }
    return fen;
}

// Materialize the supplied five-timeline fixtures. Only the central timeline
// ends at the source; the other timelines contain all six boards through 3b.
std::vector<boards_info_t> fixture(bool black, bool brawn, bool unmoved,
                                     int x, int y)
{
    std::vector<boards_info_t> result;
    for (int l = -2; l <= 2; ++l) {
        for (int t = 1; t <= 3; ++t) {
            for (bool c : {false, true}) {
                if (l == 0 && turn{t,c} > turn{2,black}) continue;
                squares b{};
                if (l == 0 && t == 2 && c == black) {
                    piece_t piece = brawn ? (black ? BRAWN_B : BRAWN_W)
                                          : (black ? PAWN_B : PAWN_W);
                    if (unmoved) piece = static_cast<piece_t>(piece | 0x80);
                    b[ppos(x,y)] = piece;
                }
                result.emplace_back(l,t,c,fen_of(b));
            }
        }
    }
    return result;
}

void place(std::vector<boards_info_t> &boards, bool c, vec4 p, piece_t piece)
{
    for (auto &[l,t,color,fen] : boards) {
        if (l != p.l() || t != p.t() || color != c) continue;
        const board position(fen);
        squares cells;
        for(int xy = 0; xy < BOARD_SIZE; ++xy) {
            const auto unmoved = (position.umove() & pmask(xy)) ? 0x80 : 0;
            cells[xy] = static_cast<piece_t>(position.get_piece(xy) | unmoved);
        }
        cells[p.xy()] = piece;
        fen = fen_of(cells);
        return;
    }
    assert(false);
}

void expect(const std::string &name, const std::vector<boards_info_t> &boards,
            bool black, vec4 source, const moves &expected)
{
    log_case(name,boards,expected.size());
    const multiverse m(timeline_layout::odd,boards);
    moves actual;
    bool duplicate = false;
    auto collect = [&](auto generator) {
        for (const auto &[tl,bb] : generator) {
            for (int xy : marked_pos(bb)) {
                const vec4 destination(xy,tl);
                duplicate |= !actual.insert(at(destination)).second;
            }
        }
    };
    if (black) collect(m.gen_moves<true>(source));
    else collect(m.gen_moves<false>(source));
    if (duplicate || actual != expected) {
        ++failures;
        std::cerr << name << (duplicate ? " duplicate destination" : " mismatch") << '\n';
        for (const auto &[x,y,t,l] : actual)
            if (!expected.contains({x,y,t,l}))
                std::cerr << "  extra " << x << ',' << y << ',' << t << ',' << l << '\n';
        for (const auto &[x,y,t,l] : expected)
            if (!actual.contains({x,y,t,l}))
                std::cerr << "  missing " << x << ',' << y << ',' << t << ',' << l << '\n';
    } else if (logging) {
        std::cout << "Test passed\n";
    }
}

moves quiet(vec4 p, bool black, bool unmoved)
{
    const int dy = black ? -1 : 1, dl = black ? 1 : -1;
    moves result{at(p+vec4(0,dy,0,0)),at(p+vec4(0,0,0,dl))};
    if (unmoved) {
        result.insert(at(p+vec4(0,2*dy,0,0)));
        result.insert(at(p+vec4(0,0,0,2*dl)));
    }
    return result;
}

void movement(bool black, bool brawn)
{
    const std::string name = std::string(black ? "black " : "white ")
        + (brawn ? "brawn " : "pawn ");
    const vec4 p(black ? 4 : 3,black ? 7 : 0,2,0);
    const int dy = black ? -1 : 1, dl = black ? 1 : -1;
    const auto original = fixture(black,brawn,true,p.x(),p.y());
    const moves baseline = quiet(p,black,true);
    expect(name+"unmoved",original,black,p,baseline);
    expect(name+"moved",fixture(black,brawn,false,p.x(),p.y()),
           black,p,quiet(p,black,false));
    const std::vector<vec4> blockers = {
        vec4(0,dy,0,0),vec4(0,2*dy,0,0),
        vec4(0,0,0,dl),vec4(0,0,0,2*dl)};
    for (int i = 0; i < 4; ++i) {
        for (piece_t blocker : {KNIGHT_W,KNIGHT_B}) {
            auto boards = original;
            place(boards,black,p+blockers[i],blocker);
            auto expected = baseline;
            expected.erase(at(p+blockers[i]));
            if (i == 0) expected.erase(at(p+blockers[1]));
            if (i == 2) expected.erase(at(p+blockers[3]));
            expect(name+"block "+std::to_string(i),boards,black,p,expected);
        }
    }
    for (int distance : {1,2}) {
        auto boards = original;
        std::erase_if(boards,[&](const boards_info_t &b) {
            return std::get<0>(b) == distance*dl && std::get<1>(b) >= 2;
        });
        auto expected = baseline;
        expected.erase(at(p+vec4(0,0,0,distance*dl)));
        if (distance == 1) expected.erase(at(p+vec4(0,0,0,2*dl)));
        expect(name+"missing board "+std::to_string(distance),boards,black,p,expected);
    }
    const vec4 edge(black ? 7 : 0,black ? 0 : 7,2,0);
    expect(name+"rank edge",fixture(black,brawn,true,edge.x(),edge.y()),
           black,edge,{at(edge+vec4(0,0,0,dl)),at(edge+vec4(0,0,0,2*dl))});
}

void captures(bool black, bool brawn, bool unmoved)
{
    const std::string name = std::string(black ? "black " : "white ")
        + (brawn ? "brawn " : "pawn ");
    const vec4 p(3,black ? 4 : 3,2,0);
    const int dy = black ? -1 : 1, dl = black ? 1 : -1;
    const piece_t enemy = black ? KNIGHT_W : KNIGHT_B;
    const piece_t friendly = black ? KNIGHT_B : KNIGHT_W;
    const auto original = fixture(black,brawn,unmoved,p.x(),p.y());
    const auto baseline = quiet(p,black,unmoved);
    // Enumerate all one-step diagonals, including forbidden directions, against
    // this explicit oracle rather than deriving answers from production code.
    moves allowed{at(vec4(-1,dy,0,0)),at(vec4(1,dy,0,0)),
                  at(vec4(0,0,-1,dl)),at(vec4(0,0,1,dl))};
    if (brawn) {
        allowed.insert(at(vec4(-1,0,0,dl)));
        allowed.insert(at(vec4(1,0,0,dl)));
        allowed.insert(at(vec4(0,dy,0,dl)));
        allowed.insert(at(vec4(0,dy,-1,0)));
    }
    moves all_expected = baseline;
    auto all_boards = original;
    int index = 0;
    for (int dx = -1; dx <= 1; ++dx)
    for (int ry = -1; ry <= 1; ++ry)
    for (int dt = -1; dt <= 1; ++dt)
    for (int rl = -1; rl <= 1; ++rl) {
        if ((dx != 0)+(ry != 0)+(dt != 0)+(rl != 0) != 2) continue;
        const vec4 d(dx,ry,dt,rl), q = p+d;
        const multiverse m(timeline_layout::odd,original);
        if (!m.inbound(q,black)) continue;
        const bool capture = allowed.contains(at(d));
        for (piece_t target : {enemy,friendly}) {
            auto boards = original;
            place(boards,black,q,target);
            auto expected = baseline;
            if (capture && target == enemy) expected.insert(at(q));
            expect(name+"diagonal "+std::to_string(index),boards,black,p,expected);
        }
        if (capture) {
            place(all_boards,black,q,enemy);
            all_expected.insert(at(q));
            if (q.tl() != p.tl()) {
                // A blocked straight timeline step must not block a diagonal.
                auto boards = original;
                place(boards,black,p+vec4(0,0,0,dl),friendly);
                place(boards,black,q,enemy);
                auto expected = baseline;
                expected.erase(at(p+vec4(0,0,0,dl)));
                expected.erase(at(p+vec4(0,0,0,2*dl)));
                expected.insert(at(q));
                expect(name+"independent capture "+std::to_string(index),
                       boards,black,p,expected);
            }
        }
        auto king_boards = original;
        place(king_boards,black,q,black ? KING_W : KING_B);
        const state s(multiverse(timeline_layout::odd,king_boards));
        const check_position view(s,-2,2);
        log_case(name+"king attack "+std::to_string(index),king_boards,capture ? 1 : 0);
        const auto checks = view.checks(black);
        if ((capture && (checks.size() != 1 || checks.front().from != p
                        || checks.front().to != q)) || (!capture && !checks.empty())) {
            ++failures;
            std::cerr << name << "incorrect king attack " << index << '\n';
        } else if (logging) {
            std::cout << "Test passed\n";
        }
        ++index;
    }
    assert(all_expected.size() == baseline.size()+(brawn ? 8 : 4));
    expect(name+"all captures",all_boards,black,p,all_expected);
    expect(name+"empty diagonals",original,black,p,baseline);
}

void file_edges(bool black, bool brawn)
{
    const int dy = black ? -1 : 1, dl = black ? 1 : -1;
    for (int x : {0,7}) {
        const vec4 p(x,black ? 4 : 3,2,0);
        auto boards = fixture(black,brawn,true,p.x(),p.y());
        auto expected = quiet(p,black,true);
        std::vector<vec4> directions{vec4(-1,dy,0,0),vec4(1,dy,0,0),
                                   vec4(0,0,-1,dl),vec4(0,0,1,dl)};
        if (brawn) {
            directions.push_back(vec4(-1,0,0,dl));
            directions.push_back(vec4(1,0,0,dl));
            directions.push_back(vec4(0,dy,0,dl));
            directions.push_back(vec4(0,dy,-1,0));
        }
        for (vec4 d : directions) {
            const vec4 q = p+d;
            if (q.outbound()) continue;
            place(boards,black,q,black ? KNIGHT_W : KNIGHT_B);
            expected.insert(at(q));
        }
        expect("file edge captures",boards,black,p,expected);
    }
}

void promotion(bool black, bool brawn)
{
    const int dy = black ? -1 : 1, dl = black ? 1 : -1;
    const piece_t original = black ? (brawn ? BRAWN_B : PAWN_B)
                                   : (brawn ? BRAWN_W : PAWN_W);
    // Approach the opponent's corner, or travel sideways from the home corner.
    for (bool home : {false,true}) {
        const vec4 p(black ? 6 : 1,home ? (black ? 7 : 0) : (black ? 1 : 6),2,0);
        std::vector<vec4> directions{vec4(0,0,0,dl)};
        if (!home) {
            directions.push_back(vec4(0,dy,0,0));
            directions.push_back(vec4(black ? 1 : -1,dy,0,0));
            if (brawn) {
                directions.push_back(vec4(0,dy,0,dl));
                directions.push_back(vec4(0,dy,-1,0));
            }
        } else if (brawn) {
            directions.push_back(vec4(black ? 1 : -1,0,0,dl));
        }
        for (vec4 d : directions) {
            const vec4 q = p+d;
            const bool physical = d.tl() == vec4(0,0,0,0);
            const bool capture = d.x() != 0 || (d.y() != 0 && !physical);
            for (bool branching : {false,true}) {
                if (physical && branching) continue;
                if (d.t() != 0 && !branching) continue;
                auto boards = fixture(black,brawn,false,p.x(),p.y());
                if (capture) place(boards,black,q,black ? KNIGHT_W : KNIGHT_B);
                if (!physical && !branching) {
                    std::erase_if(boards,[&](const boards_info_t &board) {
                        return std::get<0>(board) == q.l()
                            && turn{std::get<1>(board),std::get<2>(board)} > turn{q.t(),black};
                    });
                }
                for (auto rules : {promotion_options::QUEEN, promotion_options::KNIGHT,
                                   promotion_options::NONE}) {
                    const state before(multiverse(timeline_layout::odd,boards),rules);
                    const full_move move(p,q);
                    const bool promotes = !home && d.y() != 0 && (physical || brawn)
                        && rules != promotion_options::NONE;
                    const piece_t choice = promotes ? default_promote_to(rules) : NO_PIECE;
                    const piece_t expected = promotes ? (black ? to_black(choice) : choice) : original;
                    const auto normalized = before.normalize_promotion(move);
                    log_case("promotion application",boards,1);
                    if (!normalized || normalized->promote_to != choice) {
                        ++failures;
                        std::cerr << "incorrect promotion normalization for " << move << '\n';
                    }
                    // Check normalization through full_move and application of prepared moves.
                    for (bool prepared : {false,true}) {
                        state result = before;
                        const int line = branching ? before.new_line() : q.l();
                        const bool applied = prepared
                            ? result.apply_move(ext_move(move,choice)) : result.apply_move(move);
                        if (!applied) {
                            ++failures;
                            std::cerr << "promotion move rejected: " << move << '\n';
                            continue;
                        }
                        const auto [t,c] = result.get_timeline_end(line).get();
                        const board *destination = result.get_board_ptr(line,t,c);
                        const auto [source_t,source_c] = result.get_timeline_end(p.l()).get();
                        const board *source = result.get_board_ptr(p.l(),source_t,source_c);
                        if (!destination || !source
                            || destination->get_piece(q.xy()) != expected
                            || source->get_piece(p.xy()) != NO_PIECE) {
                            ++failures;
                            std::cerr << "incorrect promotion application for " << move
                                      << " home=" << home << " branching=" << branching
                                      << " prepared=" << prepared << '\n';
                        }
                    }
                }
            }
        }
    }
}

void en_passant(bool black, bool brawn)
{
    if(!black && !brawn)
    {
        // Turn 0w has no predecessor from which en passant can be established.
        multiverse zero(timeline_layout::odd, {{0,0,false,"8/8/8/3Pp3/8/8/8/8"}});
        const auto moves = zero.gen_physical_moves<false>(vec4(3,4,0,0));
        assert(!(moves & pmask(ppos(4,5))));
        assert(moves & pmask(ppos(3,5)));
        assert(!zero.inbound(vec4(3,4,-1,0), false));
        assert(!zero.get_board_ptr(0,-1,false));
    }
    const vec4 p(3,black ? 3 : 4,2,0);
    const int dy = black ? -1 : 1;
    const auto original = fixture(black,brawn,false,p.x(),p.y());
    for (bool victim_brawn : {false,true}) {
        for (int dx : {-1,1}) {
            const std::string victim = victim_brawn ? "brawn" : "pawn";
            auto boards = original;
            const piece_t enemy = victim_brawn ? (black ? BRAWN_W : BRAWN_B)
                                              : (black ? PAWN_W : PAWN_B);
            place(boards,black,p+vec4(dx,0,0,0),enemy);
            place(boards,black,p+vec4(dx,2*dy,-1,0),
                  static_cast<piece_t>(enemy | 0x80));
            // The immediately preceding board still has the victim at home.
            const int previous_t = black ? 2 : 1;
            const vec4 previous_home(p.x()+dx,p.y()+2*dy,previous_t,0);
            place(boards,!black,previous_home,static_cast<piece_t>(enemy | 0x80));
            const piece_t capturer = black ? (brawn ? BRAWN_B : PAWN_B)
                                          : (brawn ? BRAWN_W : PAWN_W);
            place(boards,!black,vec4(p.x(),p.y(),previous_t,0),capturer);
            const vec4 destination = p+vec4(dx,dy,0,0);
            auto expected = quiet(p,black,false);
            expected.insert(at(destination));
            expect("en passant a "+victim,boards,black,p,expected);

            auto boundary = boards;
            std::erase_if(boundary,[&](const boards_info_t &info) {
                return std::get<0>(info) == p.l()
                    && turn{std::get<1>(info),std::get<2>(info)}
                        < turn{previous_t,!black};
            });
            expect("en passant at history boundary for "+victim,
                   boundary,black,p,expected);
            std::erase_if(boundary,[&](const boards_info_t &info) {
                return std::get<0>(info) == p.l()
                    && turn{std::get<1>(info),std::get<2>(info)}
                        == turn{previous_t,!black};
            });
            expect("no preceding board for "+victim,
                   boundary,black,p,quiet(p,black,false));

            log_case("apply en passant a "+victim,boards,1);
            state s(multiverse(timeline_layout::odd,boards));
            const bool applied = s.apply_move(full_move(p,destination));
            const auto [t,c] = s.get_timeline_end(0).get();
            const board *result = s.get_board_ptr(0,t,c);
            if (!applied || !result || result->get_piece(p.xy()) != NO_PIECE
                || result->get_piece(ppos(p.x()+dx,p.y())) != NO_PIECE
                || result->get_piece(destination.xy()) != capturer) {
                ++failures;
                std::cerr << "incorrect en passant application for " << victim << '\n';
            } else if (logging) {
                std::cout << "Test passed\n";
            }

            auto blocked = boards;
            place(blocked,black,destination,black ? KNIGHT_B : KNIGHT_W);
            expect("invalid history: blocked en passant landing when capturing a "+victim,
                   blocked,black,p,quiet(p,black,false));
            auto friendly_history = boards;
            place(friendly_history,!black,previous_home,
                  static_cast<piece_t>(switch_color(enemy) | 0x80));
            expect("friendly unmoved history for "+victim,
                   friendly_history,black,p,quiet(p,black,false));
            place(boards,!black,previous_home,enemy);
            expect("no double-step history for "+victim,boards,black,p,quiet(p,black,false));
            place(boards,!black,previous_home,NO_PIECE);
            expect("absent double-step history for "+victim,boards,black,p,quiet(p,black,false));
        }
    }
}
state castling_position(const squares &pieces, int width, int height, bool black)
{
    return state(multiverse(timeline_layout::odd,
        {{0,1,black,fen_of(pieces,width,height)}}, width,height));
}

void check_castling(const state &before, vec4 p, int direction, bool allowed)
{
    const bool black = before.get_present().c();
    const auto [width,height] = before.get_board_size();
    const board &source = *before.get_board_ptr(0,1,black);
    bitboard_t castles = 0;
    for(vec4 q : before.gen_piece_move(p))
    {
        assert(q.tl() == p.tl());
        if(q.y() == p.y() && (q.x() - p.x()) * direction > 1)
            castles |= pmask(q.xy());
    }
    const int destination_x = p.x() + 2 * direction;
    assert(castles == (allowed ? pmask(ppos(destination_x,p.y())) : 0));
    if(logging)
        log_case(direction > 0 ? "right castling" : "left castling",
                 {{0,1,black,source.get_fen<true>(width,height)}}, allowed ? 1 : 0);
    if(!allowed)
    {
        if(logging) std::cout << "Test passed\n";
        return;
    }

    const full_move move(p,vec4(destination_x,p.y(),1,0));
    // Locate the first piece on this side independently of castling_rooks().
    int rook_x = p.x() + direction;
    while(source.get_piece(ppos(rook_x,p.y())) == NO_PIECE) rook_x += direction;
    assert(rook_x >= 0 && rook_x < width);
    const piece_t rook = black ? ROOK_B : ROOK_W;
    const piece_t king = black ? KING_B : KING_W;
    assert(source.get_piece(ppos(rook_x,p.y())) == rook);
    board expected = source;
    expected.set_piece(p.xy(),NO_PIECE);
    expected.set_piece(ppos(rook_x,p.y()),NO_PIECE);
    expected.set_piece(ppos(p.x()+direction,p.y()),rook);
    expected.set_piece(move.to.xy(),king);

    state after = before;
    assert(after.apply_move(move));
    const auto [t,c] = after.get_timeline_end(0).get();
    assert(after.get_board_ptr(0,t,c)->get_fen<true>(width,height)
        == expected.get_fen<true>(width,height));

    auto [info,space] = HC_info::build_HC(before);
    bool found = false;
    for(index_t axis = 0; axis < info.dimension(); ++axis)
        for(index_t coordinate : info.get_universe()[axis])
        {
            const auto cached = info.get_move_boards(axis,coordinate);
            if(!cached || cached->move != move) continue;
            assert(!cached->departure);
            assert(cached->result->get_fen<true>(width,height)
                == expected.get_fen<true>(width,height));
            found = true;
        }
    assert(found);
    if(logging) std::cout << "Test passed\n";
}

void castling_placements()
{
    for(bool black : {false,true})
        for(int width = 2; width <= 8; ++width)
            for(int rank : {0,1})
                for(int king_x = 0; king_x < width; ++king_x)
                    for(int rook_x = 0; rook_x < width; ++rook_x)
                    {
                        if(king_x == rook_x) continue;
                        const int y = black ? 3-rank : rank;
                        squares pieces{};
                        pieces[ppos(king_x,y)] = black ? KING_UB : KING_UW;
                        pieces[ppos(rook_x,y)] = black ? ROOK_UB : ROOK_UW;
                        const state before = castling_position(pieces,width,4,black);
                        const vec4 p(king_x,y,1,0);
                        const int direction = rook_x > king_x ? 1 : -1;
                        check_castling(before,p,direction,abs(rook_x-king_x) >= 2);
                        check_castling(before,p,-direction,false);
                    }
}

void castling_restrictions()
{
    for(bool black : {false,true})
    {
        const int y = black ? 6 : 1;
        const vec4 p(3,y,1,0);
        squares pieces{};
        pieces[p.xy()] = black ? KING_UB : KING_UW;
        pieces[ppos(7,y)] = black ? ROOK_UB : ROOK_UW;
        const auto friendly = [&](piece_t piece) { return black ? to_black(piece) : piece; };
        const auto enemy = [&](piece_t piece) { return black ? piece : to_black(piece); };
        for(int file : {3,4,5})
        {
            auto attacked = pieces;
            attacked[ppos(file,black ? 0 : 7)] = enemy(ROOK_W);
            check_castling(castling_position(attacked,8,8,black),p,1,false);
        }
        for(piece_t blocker : {friendly(KNIGHT_W),enemy(KNIGHT_W)})
        {
            auto blocked = pieces;
            blocked[ppos(6,y)] = blocker;
            check_castling(castling_position(blocked,8,8,black),p,1,false);
        }
        auto moved_king = pieces;
        moved_king[p.xy()] = friendly(KING_W);
        check_castling(castling_position(moved_king,8,8,black),p,1,false);
        for(piece_t rook : {friendly(ROOK_W),enemy(ROOK_UW)})
        {
            auto ineligible = pieces;
            ineligible[ppos(7,y)] = rook;
            check_castling(castling_position(ineligible,8,8,black),p,1,false);
        }
        auto nearest = pieces;
        nearest[ppos(5,y)] = friendly(ROOK_UW);
        nearest[ppos(0,y)] = friendly(ROOK_UW);
        const state multiple = castling_position(nearest,8,8,black);
        check_castling(multiple,p,1,true);
        check_castling(multiple,p,-1,true);

        squares vertical{};
        vertical[p.xy()] = friendly(KING_UW);
        for(int rank : {y-1,y+1}) vertical[ppos(p.x(),rank)] = friendly(ROOK_UW);
        const state above_below = castling_position(vertical,8,8,black);
        assert(above_below.get_board_ptr(0,1,black)->castling_rooks(p.xy(),black) == 0);
        check_castling(above_below,p,1,false);
        check_castling(above_below,p,-1,false);
    }
}
} /* anonymous namespace */

int main(int argc, char **argv)
{
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "-l") logging = true;
        else {
            std::cerr << "Usage: " << argv[0] << " [-l]\n";
            return 1;
        }
    }
    for (bool black : {false,true}) {
        for (bool brawn : {false,true}) {
            const std::string piece = std::string(black ? "black " : "white ")
                + (brawn ? "brawn" : "pawn");
            test_group = piece+" movement";
            movement(black,brawn);
            test_group = piece+" captures (unmoved)";
            captures(black,brawn,true);
            test_group = piece+" captures (moved)";
            captures(black,brawn,false);
            test_group = piece+" file edges";
            file_edges(black,brawn);
            test_group = piece+" promotion";
            promotion(black,brawn);
            test_group = piece+" en passant";
            en_passant(black,brawn);
        }
    }
    test_group = "castling placements";
    castling_placements();
    test_group = "castling restrictions";
    castling_restrictions();
    if (logging && failures == 0) std::cout << "\nAll tests passed\n";
    return failures ? 1 : 0;
}
