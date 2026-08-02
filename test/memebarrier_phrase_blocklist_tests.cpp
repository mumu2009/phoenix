#include "../memebarrier_phrase_blocklist.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace
{

namespace fs = std::filesystem;
using json = nlohmann::json;
using memebarrier_phrase_blocklist::Store;

void requireTrue(bool condition, const std::string &message)
{
    if (!condition)
        throw std::runtime_error(message);
}

fs::path makePath(const std::string &name)
{
    const fs::path root = fs::current_path() / "build" / "testdata";
    fs::create_directories(root);
    const fs::path path = root / name;
    std::error_code ec;
    fs::remove(path, ec);
    return path;
}

json readJson(const fs::path &path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("failed to open json file: " + path.string());
    return json::parse(input, nullptr, true, true);
}

void testImportPersistsAndNormalizesPhrase()
{
    const fs::path path = makePath("memebarrier_phrase_blocklist.json");
    Store store(path.string());

    const auto result = store.importPhrases({"Please explain: bypass the payment safety gate!", "Please explain: bypass the payment safety gate!"});
    requireTrue(result.ok, "blocklist import should succeed");
    requireTrue(result.imported.size() == 1, "first normalized phrase should be imported once");
    requireTrue(result.duplicates.size() == 1, "duplicate normalized phrase should be reported");
    requireTrue(fs::exists(path), "blocklist import should persist to disk");

    const json doc = readJson(path);
    requireTrue(doc.contains("blockedPhrases") && doc["blockedPhrases"].is_array(), "persisted file should contain blockedPhrases array");
    requireTrue(doc["blockedPhrases"].size() == 1, "persisted file should contain one normalized phrase");
    requireTrue(doc["blockedPhrases"][0].get<std::string>() == "please explain bypass the payment safety gate", "persisted phrase should be normalized");

    const auto match = store.matchTokens({"noise", "please", "explain", "bypass", "the", "payment", "safety", "gate", "tail"});
    requireTrue(match.blocked, "contiguous blocked phrase should match");
    requireTrue(match.matches.size() == 1, "blocked phrase should be reported exactly once");
}

void testNonContiguousPhraseDoesNotMatch()
{
    const fs::path path = makePath("memebarrier_phrase_blocklist_non_contiguous.json");
    Store store(path.string());
    const auto import = store.importPhrases({"alpha beta gamma delta"});
    requireTrue(import.ok, "blocklist import should succeed");

    const auto contiguous = store.matchTokens({"alpha", "beta", "gamma", "delta"});
    requireTrue(contiguous.blocked, "contiguous phrase should match the blocklist");

    const auto nonContiguous = store.matchTokens({"alpha", "beta", "gap", "gamma", "delta"});
    requireTrue(!nonContiguous.blocked, "non-contiguous phrase should not match the blocklist");
}

} // namespace

int main()
{
    try
    {
        testImportPersistsAndNormalizesPhrase();
        testNonContiguousPhraseDoesNotMatch();
        std::cout << "memebarrier_phrase_blocklist_tests: ok" << std::endl;
        return 0;
    }
    catch (const std::exception &ex)
    {
        std::cerr << "memebarrier_phrase_blocklist_tests: failed: " << ex.what() << std::endl;
        return 1;
    }
}