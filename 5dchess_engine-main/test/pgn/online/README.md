# Online Game Database

This collection contains 795 distinct loadable games, numbered continuously from `001.5dpgn` through `795.5dpgn`: 739 games representing the 761 populated records in [Mage's t0database](https://github.com/MageOfficial/t0database), and 56 additional games from `game.toml`.

Mage's t0database is distributed under the [upstream MIT license](LICENSE.t0database), including the original `Copyright (c) 2022 MageOChicken` notice.

The 62-game `game.toml` database was collected by **冷筱华（icemoe)** and may incorporate earlier work by others. Games represented in multiple source records share one fixture in this collection.

## Sources and variants

The t0database source is [`Data.csv` at revision `d49da4054d02fec577fd3c53368993fe453c586e`](https://github.com/MageOfficial/t0database/blob/d49da4054d02fec577fd3c53368993fe453c586e/Data.csv). Its SHA-256 is `37fb7c1ce46efb27d42a8b9d0b230df13faff1b36bef24af8c15ba5c8a6c7a4e`.

The available games use these starting variants:

| Variant | Games |
| --- | ---: |
| `Standard - Turn Zero` | 775 |
| `Standard - Defended Pawn` | 15 |
| `Standard - Two Timelines` | 3 |
| `Custom` | 2 |

[variants.json](variants.json) lists the filenames grouped by their `Variant` header for loading.

Game numbers below identify source records, independently of filenames. `Round` retains the selected source record's identifier.

Defended Pawn applies to t0database games 57, 124–129, 295–297, 313, 323, 324, 368, and 380. Two Timelines applies to t0database games 344 and 346, and `game.toml` game 27 in [026.5dpgn](026.5dpgn). These named variants use the starting boards in the engine's variant table.

The custom games are t0database games 148 and 149, in [204.5dpgn](204.5dpgn) and [205.5dpgn](205.5dpgn), use **Excessive - Two Regents** by **sanoy1997**.

## Headers and source conflicts

The t0database games use `Event "Online Game Database"` and the repository URL as `Site`. Player names, dates, original events, and links come from the source records. Player headers use the social media names `tesseract` and `cirtus` consistently. Dates use `YYYY.MM.DD`; unknown dates use `????.??.??`. Empty original events and links produce no `OriginalEvent` or `Link` header. `Public Match` and `?` indicate an unknown player.

Merged headers favor complete player names, then available dates, original events, and links. Equally complete records use the earlier source round. Missing fields are filled from other copies. Movetext comes from the copy with the most clock and other annotations; ties favor the selected header source.

Alternative player information, dates, and recorded results appear in source-labelled comment blocks immediately below the headers, for example:

```text
{
t0database/game 391 records the same game with conflicting information:
 white: Citrus
 black: Public Match
 date: 2021.01.10
 result: 1-0
}
```

The game shared by source records 391 and 597, in [436.5dpgn](436.5dpgn), uses record 597's complete player names, date, original event, and `Result "0-1"`. Its nonterminal endpoint cannot resolve the conflicting recorded outcomes. Date disagreements retain the selected source's date and preserve alternatives in comments.

## Results and game annotations

Each t0database game has exactly one `Result` header. Terminal endpoints use the independently verified result: `1-0`, `0-1`, or `1/2-1/2`. Nonterminal endpoints retain the selected source's recorded outcome, since resignations and timeouts can end games without checkmate. If neither the source nor endpoint establishes a valid result, use `*`.

Verification uses `game.get_match_status()` to examine legal actions and checks, independently of source scores and PGN result annotations. Checkmate establishes a winner, stalemate establishes a draw, and softmate remains nonterminal.

Unverified results have comments such as `{recorded:black wins}`. When source scores disagree with a terminal endpoint, comments preserve the source claim and verified winner, for example `{original data: black wins; verification: white wins}`. Game 410, in [455.5dpgn](455.5dpgn), records a verified Black win and preserves the source's invalid score pair in a comment.

Game 35, in [091.5dpgn](091.5dpgn), marks a reconstructed turn 5 supplied by a player's recollection: `(0T5)Nf3xd4 / (0T5)Nb8c6`. Its unknown clock values are omitted.

Some games preserve an incomplete final action in an `{unfinished final turn: ...}` comment, including its clocks and other annotations. Their loaded endpoint is the preceding completed action. Source records and transcription corrections are documented in [import.json](import.json).

## Endpoint identity

All 795 games have distinct complete endpoints. Duplicate detection uses the a hashing algorithm which is computed from endpoint board state and initial timeline configuration, ignoring other metadata and comments.

All games are verified to load correctly with [this project's](https://github.com/ftxi/5dchess_engine) parser.
