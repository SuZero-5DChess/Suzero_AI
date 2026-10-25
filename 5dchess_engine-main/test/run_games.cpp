#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "game.h"

std::filesystem::path executable_path(const char *name)
{
    const std::filesystem::path path = name;
    if(path.has_parent_path())
        return std::filesystem::canonical(path);

    if(const char *search_path = std::getenv("PATH"))
    {
        std::istringstream entries(search_path);
        std::string entry;
        const char separator = std::filesystem::path::preferred_separator == '\\' ? ';' : ':';
        while(std::getline(entries, entry, separator))
        {
            const auto candidate = std::filesystem::path(entry) / path;
            if(std::filesystem::is_regular_file(candidate))
                return std::filesystem::canonical(candidate);
        }
    }
    return std::filesystem::canonical(path);
}

int main(int argc, char *argv[])
{
    try
    {
        if(argc == 0 || argv[0] == nullptr || argv[0][0] == '\0')
            throw std::runtime_error("Cannot locate the run_games executable");
        const auto root = executable_path(argv[0]).parent_path().parent_path()
            / "pgn";
        std::vector<std::filesystem::path> paths;
        for(const char *collection : {"online"})
        {
            const auto directory = root / collection;
            const auto before = paths.size();
            for(const auto &entry : std::filesystem::directory_iterator(directory))
            {
                if(entry.is_regular_file() && entry.path().extension() == ".5dpgn")
                    paths.push_back(entry.path());
            }
            if(paths.size() == before)
                throw std::runtime_error("No game fixtures found in " + directory.string());
        }
        std::sort(paths.begin(), paths.end());

        std::vector<std::string> pgns;
        for(const auto &path : paths)
        {
            std::ifstream input(path);
            if(!input)
                throw std::runtime_error("Cannot open " + path.string());
            std::ostringstream contents;
            contents << input.rdbuf();
            if(input.bad())
                throw std::runtime_error("Cannot read " + path.string());
            pgns.push_back(contents.str());
        }

        const int repetitions = 1;
        const auto start = std::chrono::steady_clock::now();
        for(int iteration = 0; iteration < repetitions; ++iteration)
        {
            for(std::size_t index = 0; index < pgns.size(); ++index)
            {
                try
                {
                    // Database games can contain unresolved checks. This is an
                    // import smoke test, not an assertion of legal history.
                    game g = game::from_pgn(pgns[index], {.allow_illegal_history = true});
                }
                catch(const std::exception &error)
                {
                    throw std::runtime_error(paths[index].string() + ": " + error.what());
                }
            }
        }
        const auto end = std::chrono::steady_clock::now();
        const double duration = std::chrono::duration<double, std::milli>(end - start).count();
        const auto count = pgns.size() * repetitions;
        std::cout << "totally " << count << " games ran, took " << duration << " ms\n";
        std::cout << "average time per game: " << duration / count << " ms\n";
        std::cout << "---= run_games.cpp: all tests passed =---" << std::endl;
        return 0;
    }
    catch(const std::exception &error)
    {
        std::cerr << "run_games: " << error.what() << '\n';
        return 1;
    }
}
