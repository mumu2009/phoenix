/* product_ops.hpp - auth/monitor/module/db helpers for product surfaces */

#pragma once

#include <algorithm>
#include <system_error>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <psapi.h>
#else
#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace phoenix {
namespace product {

struct ActionResult {
    bool ok{false};
    int status{400};
    std::string error;
    std::string message;
};

inline std::string lowerCopy(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

inline std::string humanAuthError(const std::string &error)
{
    const std::string key = lowerCopy(error);
    if (key == "unauthorized" || key == "invalid credentials")
        return "用户名或密码不正确，或登录已过期。";
    if (key == "email not verified")
        return "邮箱尚未验证，请先完成验证后再登录。";
    if (key == "already bootstrapped")
        return "管理员账号已创建，请直接登录。";
    if (key == "register disabled")
        return "当前未开放自行注册，请联系管理员。";
    if (key == "missing username or password")
        return "请填写用户名和密码。";
    if (key == "missing username, password or email")
        return "请填写用户名、邮箱和密码。";
    if (key == "missing email")
        return "请填写邮箱。";
    if (key == "missing token or username/email")
        return "请填写验证码以及用户名或邮箱。";
    if (key == "missing email, token or password")
        return "请填写邮箱、重置码和新密码。";
    if (key == "missing oldpassword or newpassword")
        return "请填写旧密码和新密码。";
    if (key == "invalid username")
        return "用户名长度需为 3–32 个字符。";
    if (key == "invalid email")
        return "邮箱格式不正确。";
    if (key == "password too short")
        return "密码至少需要 6 位。";
    if (key == "username exists")
        return "该用户名已被使用。";
    if (key == "email exists")
        return "该邮箱已被使用。";
    if (key == "user not found" || key == "email not found")
        return "找不到对应账号。";
    if (key == "invalid token" || key == "reset failed")
        return "验证码无效或已过期。";
    if (key == "invalid-token")
        return "登录令牌无效，请重新登录。";
    if (key == "admin only")
        return "此操作仅管理员可用。";
    if (key == "invalid credentials")
        return "原密码不正确。";
    if (key == "new password must differ")
        return "新密码必须与旧密码不同。";
    if (key == "role not allowed")
        return "角色不合法，仅支持 user 或 admin。";
    if (key == "module forbidden")
        return "该模块不允许在此面板启停（避免误伤推理进程）。";
    if (key == "unknown module")
        return "未知逻辑模块。";
    if (key == "db missing")
        return "数据库文件尚不存在，服务首次写入后会自动创建。";
    if (key == "backup failed")
        return "备份失败，请检查备份目录权限与磁盘空间。";
    if (key.empty())
        return "操作失败。";
    return error;
}

inline ActionResult fail(int status, const std::string &error)
{
    ActionResult r;
    r.ok = false;
    r.status = status;
    r.error = error;
    r.message = humanAuthError(error);
    return r;
}

inline ActionResult okResult(int status = 200)
{
    ActionResult r;
    r.ok = true;
    r.status = status;
    return r;
}

inline bool isValidRole(const std::string &role)
{
    return role == "user" || role == "admin";
}

inline bool roleAllows(const std::string &have, const std::string &need)
{
    if (need.empty() || need == "user")
        return !have.empty();
    if (need == "admin")
        return have == "admin";
    return false;
}

inline ActionResult decideLogin(bool haveUser, bool passwordOk, bool emailVerified, bool requireVerify)
{
    if (!haveUser || !passwordOk)
        return fail(401, "Invalid credentials");
    if (requireVerify && !emailVerified)
        return fail(403, "email not verified");
    return okResult();
}

inline ActionResult decideRegister(bool allowRegister, const std::string &addUserErr)
{
    if (!allowRegister)
        return fail(403, "register disabled");
    if (!addUserErr.empty())
        return fail(400, addUserErr);
    return okResult();
}

inline ActionResult decideBootstrap(bool alreadyHasUsers, const std::string &addUserErr)
{
    if (alreadyHasUsers)
        return fail(409, "already bootstrapped");
    if (!addUserErr.empty())
        return fail(400, addUserErr);
    return okResult();
}

inline ActionResult decideAdmin(const std::string &role)
{
    if (role != "admin")
        return fail(403, "admin only");
    return okResult();
}

inline ActionResult decideSetRole(const std::string &actorRole, const std::string &newRole)
{
    ActionResult admin = decideAdmin(actorRole);
    if (!admin.ok)
        return admin;
    if (!isValidRole(newRole))
        return fail(400, "role not allowed");
    return okResult();
}

inline bool moduleIdForbidden(const std::string &id)
{
    const std::string k = lowerCopy(id);
    if (k.find("llama") != std::string::npos)
        return true;
    if (k == "supervisor" || k == "plugin-spawn" || k == "microservice")
        return true;
    return false;
}

class LogicalModuleBoard {
public:
    LogicalModuleBoard()
    {
        add_("chat-ui", "对话界面", true, false);
        add_("world", "世界模型面板", true, false);
        add_("mission", "任务面板", true, false);
        add_("study", "学习作业入口", true, false);
        add_("search", "联网搜索", true, false);
        add_("memebarrier", "内容围栏", true, false);
        add_("autonomy-loop", "自主循环（逻辑开关）", true, false);
        add_("inference-llama", "推理进程（只读：pid/端口）", true, true);
    }

    nlohmann::json listJson() const
    {
        std::lock_guard<std::mutex> lock(mu_);
        nlohmann::json arr = nlohmann::json::array();
        for (const auto &id : order_) {
            const auto &m = modules_.at(id);
            arr.push_back({{"id", m.id},
                           {"title", m.title},
                           {"enabled", m.enabled},
                           {"readonly", m.readonly},
                           {"note", m.readonly ? "只读状态，不会发送停止推理进程的命令" : ""}});
        }
        return arr;
    }

    ActionResult setEnabled(const std::string &id, bool enabled)
    {
        if (moduleIdForbidden(id) || id == "inference-llama")
            return fail(403, "module forbidden");
        std::lock_guard<std::mutex> lock(mu_);
        auto it = modules_.find(id);
        if (it == modules_.end())
            return fail(404, "unknown module");
        if (it->second.readonly)
            return fail(403, "module forbidden");
        it->second.enabled = enabled;
        return okResult();
    }

    bool enabled(const std::string &id) const
    {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = modules_.find(id);
        return it != modules_.end() && it->second.enabled;
    }

private:
    struct Mod {
        std::string id;
        std::string title;
        bool enabled{true};
        bool readonly{false};
    };

    void add_(const std::string &id, const std::string &title, bool enabled, bool readonly)
    {
        modules_[id] = Mod{id, title, enabled, readonly};
        order_.push_back(id);
    }

    mutable std::mutex mu_;
    std::unordered_map<std::string, Mod> modules_;
    std::vector<std::string> order_;
};

struct EndpointFact {
    std::string id;
    std::string title;
    std::string host;
    int port{0};
    std::int64_t pid{0};
    bool listening{false};
    std::string probe{"tcp-port"};
};

inline bool tcpPortListening(const std::string &host, int port, int timeoutMs = 200)
{
    if (host.empty() || port <= 0)
        return false;
#ifdef _WIN32
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0)
        return false;
    SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (s == INVALID_SOCKET) {
        WSACleanup();
        return false;
    }
    DWORD tv = static_cast<DWORD>(std::max(50, timeoutMs));
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char *>(&tv), sizeof(tv));
    setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, reinterpret_cast<const char *>(&tv), sizeof(tv));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<u_short>(port));
    if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    }
    int rc = connect(s, reinterpret_cast<sockaddr *>(&addr), sizeof(addr));
    closesocket(s);
    WSACleanup();
    return rc == 0;
#else
    int s = socket(AF_INET, SOCK_STREAM, 0);
    if (s < 0)
        return false;
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1)
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    timeval tv{};
    tv.tv_sec = timeoutMs / 1000;
    tv.tv_usec = (timeoutMs % 1000) * 1000;
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    int rc = connect(s, reinterpret_cast<sockaddr *>(&addr), sizeof(addr));
    close(s);
    return rc == 0;
#endif
}

inline std::uint64_t currentRssBytes()
{
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc{};
    pmc.cb = sizeof(pmc);
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        return 0;
    return static_cast<std::uint64_t>(pmc.WorkingSetSize);
#else
    std::ifstream in("/proc/self/statm");
    if (!in)
        return 0;
    std::uint64_t pages = 0;
    in >> pages >> pages;
    long page = sysconf(_SC_PAGESIZE);
    if (page <= 0)
        return 0;
    return pages * static_cast<std::uint64_t>(page);
#endif
}

inline std::int64_t currentPid()
{
#ifdef _WIN32
    return static_cast<std::int64_t>(GetCurrentProcessId());
#else
    return static_cast<std::int64_t>(getpid());
#endif
}

class RequestMeter {
public:
    void record(int status, double latencyMs)
    {
        std::lock_guard<std::mutex> lock(mu_);
        ++total_;
        if (status >= 400)
            ++errors_;
        lastLatencyMs_ = latencyMs;
        sumLatencyMs_ += latencyMs;
    }

    nlohmann::json snapshot() const
    {
        std::lock_guard<std::mutex> lock(mu_);
        nlohmann::json out;
        out["total"] = total_;
        out["errors"] = errors_;
        out["lastLatencyMs"] = lastLatencyMs_;
        out["avgLatencyMs"] = total_ ? (sumLatencyMs_ / static_cast<double>(total_)) : 0.0;
        return out;
    }

private:
    mutable std::mutex mu_;
    std::uint64_t total_{0};
    std::uint64_t errors_{0};
    double lastLatencyMs_{0};
    double sumLatencyMs_{0};
};

inline nlohmann::json buildMonitorJson(const std::vector<EndpointFact> &ends,
                                       std::uint64_t rssBytes,
                                       const nlohmann::json &requests,
                                       std::int64_t selfPid)
{
    nlohmann::json endpoints = nlohmann::json::array();
    for (const auto &e : ends) {
        nlohmann::json row = {
            {"id", e.id},
            {"title", e.title},
            {"host", e.host},
            {"port", e.port},
            {"pid", e.pid},
            {"alive", e.listening},
            {"probe", e.probe}};
        if (e.id.find("llama") != std::string::npos)
            row["healthHttpForbidden"] = true;
        endpoints.push_back(std::move(row));
    }
    return {{"ok", true},
            {"service", "phoenix-frontend"},
            {"pid", selfPid},
            {"memory", {{"rssBytes", rssBytes}, {"rssMB", rssBytes / (1024.0 * 1024.0)}}},
            {"requests", requests},
            {"endpoints", endpoints}};
}

struct DatabaseInfo {
    std::string engine;
    std::string path;
    std::string legacyDir;
    std::string backupDir;
    bool exists{false};
    bool writableParent{false};
};

inline DatabaseInfo inspectDatabase(const std::filesystem::path &dbPath,
                                    const std::filesystem::path &legacyDir,
                                    const std::filesystem::path &backupDir)
{
    DatabaseInfo info;
    info.engine = "sqlite+lmdb";
    info.path = dbPath.string();
    info.legacyDir = legacyDir.string();
    info.backupDir = backupDir.string();
    info.exists = std::filesystem::exists(dbPath);
    std::error_code ec;
    auto parent = dbPath.parent_path();
    if (parent.empty())
        parent = ".";
    info.writableParent = std::filesystem::exists(parent) || std::filesystem::create_directories(parent, ec);
    return info;
}

inline nlohmann::json databaseInfoJson(const DatabaseInfo &info)
{
    return {{"ok", true},
            {"engine", info.engine},
            {"path", info.path},
            {"legacyDir", info.legacyDir},
            {"backupDir", info.backupDir},
            {"exists", info.exists},
            {"writableParent", info.writableParent},
            {"healthy", info.exists || info.writableParent}};
}

inline std::filesystem::path suggestedBackupPath(const std::filesystem::path &dbPath,
                                                 const std::filesystem::path &backupDir,
                                                 std::int64_t epochMs)
{
    std::error_code ec;
    std::filesystem::create_directories(backupDir, ec);
    std::string name = dbPath.filename().string();
    if (name.empty())
        name = "store.sqlite";
    return backupDir / (name + "." + std::to_string(epochMs) + ".bak");
}

inline ActionResult copyDatabaseBackup(const std::filesystem::path &dbPath,
                                       const std::filesystem::path &dest)
{
    if (!std::filesystem::exists(dbPath))
        return fail(404, "db missing");
    std::error_code ec;
    std::filesystem::create_directories(dest.parent_path(), ec);
    std::filesystem::copy_file(dbPath, dest, std::filesystem::copy_options::overwrite_existing, ec);
    if (ec)
        return fail(500, "backup failed");
    ActionResult r = okResult();
    r.message = dest.string();
    return r;
}

inline bool parseHostPort(const std::string &url, std::string &host, int &port)
{
    host.clear();
    port = 0;
    auto pos = url.find("://");
    std::string rest = pos == std::string::npos ? url : url.substr(pos + 3);
    auto slash = rest.find('/');
    if (slash != std::string::npos)
        rest = rest.substr(0, slash);
    auto colon = rest.rfind(':');
    if (colon == std::string::npos)
        return false;
    host = rest.substr(0, colon);
    if (host.empty() || host == "0.0.0.0")
        host = "127.0.0.1";
    const std::string portText = rest.substr(colon + 1);
    if (portText.empty() || !std::all_of(portText.begin(), portText.end(), [](unsigned char c) {
            return std::isdigit(c);
        }))
        return false;
    port = std::stoi(portText);
    return port > 0;
}

inline std::int64_t nowMs()
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

} // namespace product
} // namespace phoenix
