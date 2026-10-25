#include "board.h"
#include <algorithm>
#include <cctype>
#include <string>
#include <iostream>
#include <sstream>
#include <utility>

#include "magic.h"

board::board(std::string fen, int size_x, int size_y) : bbs{}, umove_mask{0}
{
    if(size_x < 1 || size_x > BOARD_LENGTH || size_y < 1 || size_y > BOARD_LENGTH)
        throw std::invalid_argument("Board dimensions must be between 1 and 8");

    using error = board_fen_error;
    if(std::all_of(fen.begin(), fen.end(), [](unsigned char c) { return std::isspace(c); }))
        throw board_fen_error(
            error::too_few_ranks{{static_cast<std::size_t>(size_y), 0}},
            fen, fen.size(), fen.size());

    bbs[WHITE] = bbs[BLACK] = ~bitboard_t{0};
    std::size_t rank_start = 0;
    char c;
    int x = 0, y = size_y - 1;
    for(size_t i = 0; i < fen.length(); i++)
    {
        c = fen[i];
        if(i+1 < fen.length() && fen[i+1] == '*')
        {
            if(x >= size_x) // check overflow
            {
                throw board_fen_error(
                    error::rank_too_long{{static_cast<std::size_t>(size_x), static_cast<std::size_t>(x + 1)}},
                    fen, i, i + 2);
            }
            switch(c)
            {
                case 'K':
                case 'R':
                case 'P':
                case 'W':
                case 'k':
                case 'r':
                case 'p':
                case 'w':
                    set_piece(ppos(x,y), static_cast<piece_t>(c));
                    umove_mask |= pmask(ppos(x,y));
                    break;
                default:
                    throw board_fen_error(error::invalid_unmoved_marker{}, fen, i, i + 2);
            }
            i++; //so that this loop increment i by 2, skipping c and the following *
            x++;
        }
        else if(c == '/')
        {
            if(x < size_x)
                throw board_fen_error(
                    error::rank_too_short{{static_cast<std::size_t>(size_x), static_cast<std::size_t>(x)}},
                    fen, rank_start, i);
            y--;
            if(y < 0)
            {
                throw board_fen_error(error::too_many_ranks{{static_cast<std::size_t>(size_y),
                        static_cast<std::size_t>(std::count(fen.begin(), fen.end(), '/')) + 1}},
                    fen, i, i + 1);
            }
            x = 0;
            rank_start = i + 1;
        }
        else if(std::isspace(static_cast<unsigned char>(c)))
        {
            //pass
        }
        else
        {
            int shift = 1;
            if(std::isdigit(static_cast<unsigned char>(c)))
                shift = c - '0';
            if(x + shift > size_x) // check overflow
            {
                throw board_fen_error(
                    error::rank_too_long{{static_cast<std::size_t>(size_x), static_cast<std::size_t>(x + shift)}},
                    fen, i, i + 1);
            }
            switch(c)
            {
                case '#':
                    this->set_piece(ppos(x,y), WALL_PIECE); x++;
                    break;
                case '9':
                    this->set_piece(ppos(x,y), NO_PIECE); x++; [[fallthrough]];
                case '8':
                    this->set_piece(ppos(x,y), NO_PIECE); x++; [[fallthrough]];
                case '7':
                    this->set_piece(ppos(x,y), NO_PIECE); x++; [[fallthrough]];
                case '6':
                    this->set_piece(ppos(x,y), NO_PIECE); x++; [[fallthrough]];
                case '5':
                    this->set_piece(ppos(x,y), NO_PIECE); x++; [[fallthrough]];
                case '4':
                    this->set_piece(ppos(x,y), NO_PIECE); x++; [[fallthrough]];
                case '3':
                    this->set_piece(ppos(x,y), NO_PIECE); x++; [[fallthrough]];
                case '2':
                    this->set_piece(ppos(x,y), NO_PIECE); x++; [[fallthrough]];
                case '1':
                    this->set_piece(ppos(x,y), NO_PIECE); x++; [[fallthrough]];
                case '0':
                    break;
                case 'K':
                case 'Q':
                case 'B':
                case 'N':
                case 'R':
                case 'P':
                case 'U':
                case 'D':
                case 'W':
                case 'S':
                case 'Y':
                case 'C':
                case 'k':
                case 'q':
                case 'b':
                case 'n':
                case 'r':
                case 'p':
                case 'u':
                case 'd':
                case 'w':
                case 's':
                case 'y':
                case 'c':
                    this->set_piece(ppos(x,y), static_cast<piece_t>(c)); x++;
                    break;
                case '*':
                    throw board_fen_error(error::invalid_unmoved_marker{}, fen, i, i + 1);
                default:
                    throw board_fen_error(error::invalid_piece{}, fen, i, i + 1);
            }
        }
    }
    if(x < size_x)
        throw board_fen_error(
            error::rank_too_short{{static_cast<std::size_t>(size_x), static_cast<std::size_t>(x)}},
            fen, rank_start, fen.size());
    if(y > 0)
        throw board_fen_error(
            error::too_few_ranks{{static_cast<std::size_t>(size_y), static_cast<std::size_t>(size_y - y)}},
            fen, fen.size(), fen.size());
}


piece_t board::get_piece(int pos) const
{
    piece_t piece;
    bitboard_t z = pmask(pos);
    if(z & bbs[WHITE])
    {
        if(z & king())
            piece = KING_W;
        else if(z & common_king())
            piece = COMMON_KING_W;
        else if(z & queen())
            piece = QUEEN_W;
        else if(z & royal_queen())
            piece = ROYAL_QUEEN_W;
        else if(z & bishop())
            piece = BISHOP_W;
        else if(z & knight())
            piece = KNIGHT_W;
        else if(z & rook())
            piece = ROOK_W;
        else if(z & pawn())
            piece = PAWN_W;
        else if(z & unicorn())
            piece = UNICORN_W;
        else if(z & dragon())
            piece = DRAGON_W;
        else if(z & brawn())
            piece = BRAWN_W;
        else if(z & princess())
            piece = PRINCESS_W;
        else if(z & wall())
            piece = WALL_PIECE;
        else
            throw std::runtime_error("get_piece: unknown piece\n");
    }
    else if(z & bbs[BLACK])
    {
        if(z & king())
            piece = KING_B;
        else if(z & common_king())
            piece = COMMON_KING_B;
        else if(z & queen())
            piece = QUEEN_B;
        else if(z & royal_queen())
            piece = ROYAL_QUEEN_B;
        else if(z & bishop())
            piece = BISHOP_B;
        else if(z & knight())
            piece = KNIGHT_B;
        else if(z & rook())
            piece = ROOK_B;
        else if(z & pawn())
            piece = PAWN_B;
        else if(z & unicorn())
            piece = UNICORN_B;
        else if(z & dragon())
            piece = DRAGON_B;
        else if(z & brawn())
            piece = BRAWN_B;
        else if(z & princess())
            piece = PRINCESS_B;
        else
            throw std::runtime_error("board::get_piece: unknown piece\n");
    }
    else
    {
        piece = NO_PIECE;
    }
    return piece;
}

void board::set_piece(int pos, piece_t p)
{
    bitboard_t z = pmask(pos);
    umove_mask &= ~z;
    for(int i = 0; i < BBS_INDICES_COUNT; i++)
    {
        bbs[i] &= ~z;
    }
    if(p == NO_PIECE)
    {
        //pass
    }
    else if(p == WALL_PIECE)
    {
        bbs[board::WHITE] |= z;
        bbs[board::BLACK] |= z;
    }
    else
    {
        if(piece_color(p)==0)
        {
            bbs[board::WHITE] |= z;
        }
        else
        {
            bbs[board::BLACK] |= z;
        }
        switch(to_white(piece_name(p)))
        {
            case KING_W:
                bbs[board::ROYAL] |= z;
                [[fallthrough]];
            case COMMON_KING_W:
                bbs[board::LKING] |= z;
                break;
            case ROOK_W:
                bbs[board::LROOK] |= z;
                break;
            case BISHOP_W:
                bbs[board::LBISHOP] |= z;
                break;
            case UNICORN_W:
                bbs[board::LUNICORN] |= z;
                break;
            case DRAGON_W:
                bbs[board::LDRAGON] |= z;
                break;
            case ROYAL_QUEEN_W:
                bbs[board::ROYAL] |= z;
                [[fallthrough]];
            case QUEEN_W:
                bbs[board::LROOK] |= z;
                bbs[board::LBISHOP] |= z;
                bbs[board::LUNICORN] |= z;
                bbs[board::LDRAGON] |= z;
                break;
            case PRINCESS_W:
                bbs[board::LROOK] |= z;
                bbs[board::LBISHOP] |= z;
                break;
            case KNIGHT_W:
                bbs[board::LKNIGHT] |= z;
                break;
            case BRAWN_W:
                bbs[board::LRAWN] |= z;
                [[fallthrough]];
            case PAWN_W:
                bbs[board::LPAWN] |= z;
                break;
            default:
                std::ostringstream oss;
                oss << "board::set_piece:" << p << "not implemented" << std::endl;
                throw std::runtime_error(oss.str());
                break;
        }
    }
}

std::shared_ptr<const board> board::replace_piece(int pos, piece_t p) const
{
    std::shared_ptr<board> b_ptr = std::make_shared<board>(*this);
    b_ptr->set_piece(pos, p);
    return b_ptr;
}

std::shared_ptr<const board> board::move_piece(int from, int to) const
{
    std::shared_ptr<board> b_ptr = std::make_shared<board>(*this);
    b_ptr->set_piece(to, get_piece(from));
    b_ptr->set_piece(from, NO_PIECE);
    return b_ptr;
}

std::string board::to_string() const
{
    std::string result;
    for(int y = BOARD_LENGTH - 1; y >= 0; --y)
    {
        for(int x = 0; x < BOARD_LENGTH; ++x)
        {
            const piece_t p = get_piece(ppos(x,y));
            if(p == NO_PIECE)
                result += ". ";
            else if(p == WALL_PIECE)
                result += "##";
            else
            {
                result += static_cast<char>(p);
                result += ' ';
            }
        }
        result += '\n';
    }
    return result;
}

template<bool SHOW_UMOVE>
std::string board::get_fen(int size_x, int size_y) const
{
    if(size_x < 1 || size_x > BOARD_LENGTH || size_y < 1 || size_y > BOARD_LENGTH)
        throw std::invalid_argument("get_fen(): Board dimensions must be between 1 and 8");
    std::string result = "/";
    for (int y = size_y - 1; y >= 0; y--)
    {
        for (int x = 0; x < size_x; x++)
        {
            switch(this->get_piece(ppos(x,y)))
            {
                case NO_PIECE:
                    if(isdigit(result.back()))
                    {
                        result.back() += 1;
                    }
                    else
                    {
                        result += '1';
                    }
                    break;
                case WALL_PIECE:
                    result += '#';
                    break;
                default:
                    result += get_piece(ppos(x,y));
                    if constexpr (SHOW_UMOVE)
                    {
                        if(umove() & pmask(ppos(x, y)))
                        {
                            result += "*";
                        }
                    }
                    break;
            }
        }
        result += "/";
    }
    result.erase(result.begin());
    result.pop_back(); // remove the extra '/'
    return result;
}

bitboard_t board::castling_rooks(int pos, bool color) const noexcept
{
    const bitboard_t z = pmask(pos);
    const bitboard_t vertical_blockers = shift_north(z) | shift_south(z);
    const bitboard_t friendly = color ? black() : white();
    return rook_attack(pos, occupied() | vertical_blockers)
        & friendly & rook() & umove() & ~vertical_blockers;
}

bitboard_t board::attacks_to(int pos) const noexcept
{
    bitboard_t w = white(), b = black();
    bitboard_t pawns = lpawn(), all = w | b;
    bitboard_t white_pawns = pawns & w;
    bitboard_t black_pawns = pawns & b;
    return (white_pawn_attack(pos) & black_pawns)
        |  (black_pawn_attack(pos) & white_pawns)
        |  (king_attack(pos)       & lking())
        |  (knight_attack(pos)     & lknight())
        |  (rook_attack(pos, all)  & lrook())
        |  (bishop_attack(pos, all)& lbishop());
}

bool board::is_under_attack(int pos, int color) const noexcept
{
    bitboard_t hostile = color ? white() : black();
    bitboard_t pawns = lpawn();
    if(color == 0 && white_pawn_attack(pos) & pawns & hostile)
        return true;
    if(color != 0 && black_pawn_attack(pos) & pawns & hostile)
        return true;
    if(king_attack(pos) & lking() & hostile)
        return true;
    if(knight_attack(pos) & lknight() & hostile)
        return true;
    bitboard_t all = white() | black();
    if(bishop_attack(pos, all) & lbishop() & hostile)
        return true;
    if(rook_attack(pos, all) & lrook() & hostile)
        return true;
    return false;
}


template std::string board::get_fen<true>(int, int) const;
template std::string board::get_fen<false>(int, int) const;
