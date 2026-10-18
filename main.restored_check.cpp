
#include <drogon/drogon.h>
#include <nlohmann/json.hpp>
#include <jwt-cpp/jwt.h>
#include <jwt-cpp/traits/nlohmann-json/traits.h>
#include <json/json.h>
#include <trantor/utils/Utilities.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <deque>
#include <cstring>
#include <future>
#include <iomanip>
#include <functional>
#include <iostream>
#include <list>
#include <limits>
#include <map>
#include <mutex>
#include <queue>
#include <optional>
#include <random>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <csignal>
#include <exception>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "transformer.hpp"
#include "transformer_main.cpp"
#include "model_lifecycle.hpp"
#include "autonomy_stack.hpp"
#include "addon.hpp"
#include "loggerCXXH.hpp"
#include "DATABASE_079.hpp"
#include "frontend_server.hpp"
#include "module_mount.hpp"

#ifndef AI_OUTSIDES_LINK_MODE
#define AI_OUTSIDES_LINK_MODE "headers-only"
#endif

extern "C" {
int transformer_ollama_ft_bridge_set_module(const char *module_name, char *error_buf, size_t error_cap);
int transformer_ollama_ft_bridge_init(char *error_buf, size_t error_cap);
int transformer_ollama_ft_bridge_run_main(int argc, const char *const *argv, char *error_buf, size_t error_cap);
int transformer_ollama_ft_bridge_train(const char *append_corpus,
                                       const char *output_dir,
                                       const char *ollama_model,
                                       const char *hf_model,
                                       int self_play_pairs,
                                       int epochs,
                                       double lr,
                                       char *error_buf,
                                       size_t error_cap);
void transformer_ollama_ft_bridge_finalize(void);
}

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "psapi.lib")
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <fcntl.h>
#include <signal.h>
#include <unistd.h>
#include <sys/resource.h>
#endif

#ifdef HAVE_REDIS
#include <sw/redis++/redis++.h>
#endif

#ifdef HAVE_LMDB
#include <lmdb.h>
#endif

#ifdef HAVE_GUMBO
#include <gumbo.h>
#endif

#ifdef HAVE_POPPLER
#include <poppler-document.h>
#include <poppler-page.h>
#endif

#ifdef HAVE_CURL
#include <curl/curl.h>
#endif

#ifdef HAVE_TORCH
#include <torch/torch.h>
#endif

using json = nlohmann::json;
namespace fs = std::filesystem;

// 将 JsonCpp 的 Json::Value 递归转换为 nlohmann::json。
// 调用方式：当外部模块返回 JsonCpp 对象时通过本函数桥接。
// 实现思路：按 ValueType 分支递归处理对象与数组。
// 注意事项：未知类型会回退为 null，调用方需做容错。
static json fromJsoncpp(const Json::Value &v)
{
    switch (v.type())
    {
    case Json::nullValue:
        return nullptr;
    case Json::intValue:
        return (int64_t)v.asInt64();
    case Json::uintValue:
        return (uint64_t)v.asUInt64();
    case Json::realValue:
        return v.asDouble();
    case Json::stringValue:
        return v.asString();
    case Json::booleanValue:
        return v.asBool();
    case Json::arrayValue:
    {
        json out = json::array();
        for (const auto &item : v)
            out.push_back(fromJsoncpp(item));
        return out;
    }
    case Json::objectValue:
    {
        json out = json::object();
        for (auto it = v.begin(); it != v.end(); ++it)
        {
            out[it.name()] = fromJsoncpp(*it);
        }
        return out;
    }
    default:
        return nullptr;
    }
}

// 将 nlohmann::json 递归转换为 JsonCpp 的 Json::Value。
// 调用方式：需要与 JsonCpp API 交互时调用。
// 实现思路：按 json 类型分派并递归构建数组/对象。
// 注意事项：数值类型会按整型/浮点分支转换。
static Json::Value toJsoncpp(const json &v)
{
    if (v.is_null())
        return Json::Value();
    if (v.is_boolean())
        return Json::Value(v.get<bool>());
    if (v.is_number_integer())
        return Json::Value((Json::Int64)v.get<long long>());
    if (v.is_number_unsigned())
        return Json::Value((Json::UInt64)v.get<unsigned long long>());
    if (v.is_number_float())
        return Json::Value(v.get<double>());
    if (v.is_string())
        return Json::Value(v.get<std::string>());
    if (v.is_array())
    {
        Json::Value arr(Json::arrayValue);
        for (const auto &item : v)
            arr.append(toJsoncpp(item));
        return arr;
    }
    if (v.is_object())
    {
        Json::Value obj(Json::objectValue);
        for (auto it = v.begin(); it != v.end(); ++it)
            obj[it.key()] = toJsoncpp(it.value());
        return obj;
    }
    return Json::Value();
}

static std::string g_selfPath;

static std::mutex gFrontendProcMu;
static std::mutex gPidFileMu;
static fs::path gPidFilePath = fs::path("runtime_store") / "phoenix_main.pid";
static bool gPidFileArmed = false;
#ifdef _WIN32
static HANDLE gFrontendProcessHandle = nullptr;
static DWORD gFrontendProcessId = 0;
static HANDLE gFrontendJobHandle = nullptr;
static HANDLE gPidFileHandle = INVALID_HANDLE_VALUE;
static std::atomic<bool> gCtrlForceExitArmed{false};
static std::atomic<DWORD> gCtrlForceExitMs{12000};

static DWORD WINAPI forceExitWatchdogThread(LPVOID) noexcept
{
    DWORD waitMs = gCtrlForceExitMs.load(std::memory_order_relaxed);
    if (waitMs < 1000)
        waitMs = 1000;
    if (waitMs > 600000)
        waitMs = 600000;
    Sleep(waitMs);
    TerminateProcess(GetCurrentProcess(), 130);
    return 0;
}

static void armCtrlForceExitWatchdog() noexcept
{
    bool expected = false;
    if (!gCtrlForceExitArmed.compare_exchange_strong(expected, true))
        return;
    HANDLE h = CreateThread(nullptr, 0, forceExitWatchdogThread, nullptr, 0, nullptr);
    if (h)
        CloseHandle(h);
}

static void ensureFrontendJobObjectLocked() noexcept
{
    if (gFrontendJobHandle)
        return;
    HANDLE job = CreateJobObjectA(nullptr, nullptr);
    if (!job)
        return;
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION info{};
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    if (!SetInformationJobObject(job, JobObjectExtendedLimitInformation, &info, sizeof(info)))
    {
        CloseHandle(job);
        return;
    }
    gFrontendJobHandle = job;
}
#else
static pid_t gFrontendProcessPid = -1;
#endif

static void appendFatalLog(const std::string &line);

static unsigned long long currentProcessId() noexcept
{
#ifdef _WIN32
    return static_cast<unsigned long long>(GetCurrentProcessId());
#else
    return static_cast<unsigned long long>(getpid());
#endif
}

static bool processIsRunning(unsigned long long pid) noexcept
{
    if (pid == 0)
        return false;
#ifdef _WIN32
    HANDLE h = OpenProcess(SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION, FALSE, static_cast<DWORD>(pid));
    if (!h)
        return false;
    DWORD waitCode = WaitForSingleObject(h, 0);
    CloseHandle(h);
    return waitCode == WAIT_TIMEOUT;
#else
    return kill(static_cast<pid_t>(pid), 0) == 0;
#endif
}

static unsigned long long readPidFileValue(const fs::path &pidPath) noexcept
{
    try
    {
        std::ifstream in(pidPath);
        unsigned long long pid = 0;
        in >> pid;
        return pid;
    }
    catch (...)
    {
        return 0;
    }
}

static void cleanupPidFile() noexcept
{
    std::lock_guard<std::mutex> lock(gPidFileMu);
    if (!gPidFileArmed)
        return;
#ifdef _WIN32
    if (gPidFileHandle != INVALID_HANDLE_VALUE)
    {
        CloseHandle(gPidFileHandle);
        gPidFileHandle = INVALID_HANDLE_VALUE;
    }
#endif
    std::error_code ec;
    fs::remove(gPidFilePath, ec);
    gPidFileArmed = false;
}

static bool acquirePidFile(const fs::path &pidPath, std::string &errorOut)
{
    std::lock_guard<std::mutex> lock(gPidFileMu);
    gPidFilePath = pidPath;
    std::error_code ec;
    fs::create_directories(gPidFilePath.parent_path(), ec);

#ifdef _WIN32
    auto tryCreate = [&]() -> bool
    {
        gPidFileHandle = CreateFileW(gPidFilePath.wstring().c_str(),
                                     GENERIC_READ | GENERIC_WRITE,
                                     FILE_SHARE_READ,
                                     nullptr,
                                     CREATE_NEW,
                                     FILE_ATTRIBUTE_NORMAL,
                                     nullptr);
        return gPidFileHandle != INVALID_HANDLE_VALUE;
    };

    if (!tryCreate())
    {
        DWORD code = GetLastError();
        if (code == ERROR_FILE_EXISTS || code == ERROR_ALREADY_EXISTS)
        {
            unsigned long long existingPid = readPidFileValue(gPidFilePath);
            if (processIsRunning(existingPid))
            {
                errorOut = "pid file exists and process is running, pid=" + std::to_string(existingPid) + " file=" + gPidFilePath.string();
                return false;
            }
            fs::remove(gPidFilePath, ec);
            if (!tryCreate())
            {
                errorOut = "pid file create failed after stale cleanup, file=" + gPidFilePath.string();
                return false;
            }
        }
        else
        {
            errorOut = "pid file create failed, code=" + std::to_string(static_cast<unsigned long long>(code)) + " file=" + gPidFilePath.string();
            return false;
        }
    }

    std::string body = std::to_string(currentProcessId()) + "\n";
    DWORD written = 0;
    BOOL ok = WriteFile(gPidFileHandle, body.data(), static_cast<DWORD>(body.size()), &written, nullptr);
    FlushFileBuffers(gPidFileHandle);
    if (!ok || written != body.size())
    {
        CloseHandle(gPidFileHandle);
        gPidFileHandle = INVALID_HANDLE_VALUE;
        fs::remove(gPidFilePath, ec);
        errorOut = "pid file write failed, file=" + gPidFilePath.string();
        return false;
    }
    gPidFileArmed = true;
    return true;
#else
    int fd = ::open(gPidFilePath.string().c_str(), O_CREAT | O_EXCL | O_WRONLY, 0644);
    if (fd < 0)
    {
        unsigned long long existingPid = readPidFileValue(gPidFilePath);
        if (processIsRunning(existingPid))
        {
            errorOut = "pid file exists and process is running, pid=" + std::to_string(existingPid) + " file=" + gPidFilePath.string();
            return false;
        }
        fs::remove(gPidFilePath, ec);
        fd = ::open(gPidFilePath.string().c_str(), O_CREAT | O_EXCL | O_WRONLY, 0644);
        if (fd < 0)
        {
            errorOut = "pid file create failed, file=" + gPidFilePath.string();
            return false;
        }
    }
    std::string body = std::to_string(currentProcessId()) + "\n";
    ssize_t n = ::write(fd, body.data(), body.size());
    ::close(fd);
    if (n != static_cast<ssize_t>(body.size()))
    {
        fs::remove(gPidFilePath, ec);
        errorOut = "pid file write failed, file=" + gPidFilePath.string();
        return false;
    }
    gPidFileArmed = true;
    return true;
#endif
}

static void cleanupFrontendProcess() noexcept
{
    std::lock_guard<std::mutex> lock(gFrontendProcMu);
#ifdef _WIN32
    if (gFrontendProcessHandle)
    {
        DWORD waitCode = WaitForSingleObject(gFrontendProcessHandle, 0);
        if (waitCode == WAIT_TIMEOUT)
        {
            TerminateProcess(gFrontendProcessHandle, 0);
            WaitForSingleObject(gFrontendProcessHandle, 1500);
        }
        CloseHandle(gFrontendProcessHandle);
        gFrontendProcessHandle = nullptr;
        gFrontendProcessId = 0;
    }
    if (gFrontendJobHandle)
    {
        CloseHandle(gFrontendJobHandle);
        gFrontendJobHandle = nullptr;
    }
#else
    if (gFrontendProcessPid > 0)
    {
        kill(gFrontendProcessPid, SIGTERM);
        gFrontendProcessPid = -1;
    }
#endif
}

#ifdef _WIN32
static BOOL WINAPI onConsoleControl(DWORD ctrlType)
{
    switch (ctrlType)
    {
    case CTRL_C_EVENT:
    case CTRL_BREAK_EVENT:
    case CTRL_CLOSE_EVENT:
    case CTRL_LOGOFF_EVENT:
    case CTRL_SHUTDOWN_EVENT:
        armCtrlForceExitWatchdog();
        cleanupFrontendProcess();
        cleanupPidFile();
        return FALSE;
    default:
        return FALSE;
    }
}

static LONG WINAPI onUnhandledException(EXCEPTION_POINTERS *exceptionInfo)
{
    try
    {
        DWORD code = exceptionInfo && exceptionInfo->ExceptionRecord ? exceptionInfo->ExceptionRecord->ExceptionCode : 0;
        std::ostringstream oss;
        oss << "windows unhandled exception code=0x" << std::hex << (unsigned long)code;
        appendFatalLog(oss.str());
    }
    catch (...)
    {
    }
    return EXCEPTION_EXECUTE_HANDLER;
}
#endif

static std::mutex gFatalLogMu;
static fs::path gFatalLogPath = fs::path("runtime_store") / "logs" / "fatal_runtime.log";

static void appendFatalLog(const std::string &line)
{
    try
    {
        std::lock_guard<std::mutex> lock(gFatalLogMu);
        fs::create_directories(gFatalLogPath.parent_path());
        std::ofstream out(gFatalLogPath, std::ios::app);
        if (!out)
            return;
        auto now = std::chrono::system_clock::now();
        auto t = std::chrono::system_clock::to_time_t(now);
        out << std::put_time(std::localtime(&t), "%Y-%m-%d %H:%M:%S") << " | " << line << "\n";
    }
    catch (...)
    {
    }
}

static void onFatalTerminate()
{
    std::string msg = "std::terminate";
    try
    {
        auto ep = std::current_exception();
        if (ep)
        {
            try
            {
                std::rethrow_exception(ep);
            }
            catch (const std::exception &e)
            {
                msg += std::string(": ") + e.what();
            }
            catch (...)
            {
                msg += ": non-std exception";
            }
        }
    }
    catch (...)
    {
    }
    appendFatalLog(msg);
    cleanupPidFile();
    std::_Exit(134);
}

static void onFatalSignal(int signum)
{
    appendFatalLog(std::string("fatal signal=") + std::to_string(signum));
    cleanupPidFile();
    std::_Exit(128 + signum);
}

static void installFatalHooks(const fs::path &baseDir)
{
    try
    {
        gFatalLogPath = baseDir / "logs" / "fatal_runtime.log";
    }
    catch (...)
    {
    }
    std::set_terminate(onFatalTerminate);
    std::signal(SIGABRT, onFatalSignal);
    std::signal(SIGSEGV, onFatalSignal);
    std::signal(SIGILL, onFatalSignal);
    std::signal(SIGFPE, onFatalSignal);
    std::signal(SIGTERM, onFatalSignal);
}

// 读取环境变量并提供回退值。
// 调用方式：传入环境变量名和可选默认值。
// 实现思路：调用 std::getenv，缺失时返回 fallback。
// 注意事项：返回的是复制后的 std::string，可安全持有。
static std::string getEnv(const std::string &key, const std::string &fallback = "")
{
    const char *v = std::getenv(key.c_str());
    if (!v)
        return fallback;
    return std::string(v);
}

static void setRuntimeEnvValue(const char *key, const std::string &value)
{
#ifdef _WIN32
    SetEnvironmentVariableA(key, value.c_str());
#else
    setenv(key, value.c_str(), 1);
#endif
}

static void setRuntimeEnvIfArgPresent(const std::map<std::string, std::string> &args,
                                      const std::string &argKey,
                                      const char *envKey)
{
    auto it = args.find(argKey);
    if (it != args.end() && !it->second.empty())
    {
        setRuntimeEnvValue(envKey, it->second);
    }
}

static void applyFrontendEnvFromArgs(const std::map<std::string, std::string> &args)
{
    setRuntimeEnvIfArgPresent(args, "frontend-enabled", "AI_FRONTEND_ENABLED");
    setRuntimeEnvIfArgPresent(args, "frontend-http-log", "FRONTEND_HTTP_LOG");
    setRuntimeEnvIfArgPresent(args, "frontend-chat-queue-wait-ms", "FRONTEND_CHAT_QUEUE_WAIT_MS");
    setRuntimeEnvIfArgPresent(args, "frontend-chat-upstream-timeout-ms", "FRONTEND_CHAT_UPSTREAM_TIMEOUT_MS");
    setRuntimeEnvIfArgPresent(args, "frontend-api-upstream-timeout-ms", "FRONTEND_API_UPSTREAM_TIMEOUT_MS");
    setRuntimeEnvIfArgPresent(args, "frontend-chat-max-inflight", "FRONTEND_CHAT_MAX_INFLIGHT");
    setRuntimeEnvIfArgPresent(args, "frontend-port", "FRONTEND_PORT");

    auto syncJwtSecret = [&](const std::string &value)
    {
        if (value.empty())
            return;
        setRuntimeEnvValue("AI_AUTH_JWT_SECRET", value);
        setRuntimeEnvValue("AUTH_JWT_SECRET", value);
        setRuntimeEnvValue("JWT_SECRET", value);
    };
    auto itAuthJwt = args.find("auth-jwt-secret");
    if (itAuthJwt != args.end() && !itAuthJwt->second.empty())
        syncJwtSecret(itAuthJwt->second);
    auto itJwt = args.find("jwt-secret");
    if (itJwt != args.end() && !itJwt->second.empty())
        syncJwtSecret(itJwt->second);

    auto itStudyPort = args.find("study-port");
    if (itStudyPort != args.end() && !itStudyPort->second.empty())
        setRuntimeEnvValue("FRONTEND_PORT", itStudyPort->second);

    std::string host = getEnv("AI_GATEWAY_HOST", "127.0.0.1");
    std::string port = getEnv("CONTROLLER_PORT", "5080");
    auto itHost = args.find("gateway-host");
    if (itHost != args.end() && !itHost->second.empty())
        host = itHost->second;
    auto itPort = args.find("port");
    if (itPort != args.end() && !itPort->second.empty())
        port = itPort->second;
    setRuntimeEnvValue("AI_API_BASE", std::string("http://") + host + ":" + port);
}

struct Config;
static void applyFrontendEnvFromConfig(const Config &config);

// 将字符串解析为布尔值（宽松规则）。
// 调用方式：读取配置项后转换成布尔开关。
// 实现思路：先转小写，再对常见 false 字面量做否定判断。
// 注意事项：非 false 关键字均视为 true。
static bool boolFrom(const std::string &value, bool fallback = true)
{
    std::string v = value;
    auto begin = v.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos)
        return fallback;
    auto end = v.find_last_not_of(" \t\r\n");
    v = v.substr(begin, end - begin + 1);
    if (v.empty())
        return fallback;
    std::transform(v.begin(), v.end(), v.begin(), ::tolower);
    if (v == "1" || v == "true" || v == "on" || v == "yes")
        return true;
    if (v == "0" || v == "false" || v == "off" || v == "no")
        return false;
    return fallback;
}

static int roughTokenCount(const std::string &s)
{
    int c = 0;
    bool in = false;
    for (unsigned char ch : s)
    {
        bool sp = std::isspace(ch) != 0;
        if (!sp && !in)
        {
            c++;
            in = true;
        }
        else if (sp)
        {
            in = false;
        }
    }
    return c;
}

// 复制并裁剪字符串首尾空白字符。
// 调用方式：用于数值解析或文本预处理前标准化输入。
// 实现思路：定位首尾非空白下标后执行 substr。
// 注意事项：全空白字符串会返回空串。
static std::string trimCopy(const std::string &s)
{
    auto start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos)
        return "";
    auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

// 按 JavaScript 风格解析数字文本。
// 调用方式：传入字符串和输出引用，成功返回 true。
// 实现思路：支持 NaN/Infinity 及 0x/0b/0o 前缀，再回退 stod。
// 注意事项：解析失败返回 false，调用方需自行处理默认值。
static bool parseNumberLikeJs(const std::string &s, double &out)
{
    std::string t = trimCopy(s);
    if (t.empty())
    {
        out = 0.0;
        return true;
    }
    if (t == "NaN" || t == "+NaN" || t == "-NaN")
    {
        out = std::numeric_limits<double>::quiet_NaN();
        return true;
    }
    if (t == "Infinity" || t == "+Infinity")
    {
        out = std::numeric_limits<double>::infinity();
        return true;
    }
    if (t == "-Infinity")
    {
        out = -std::numeric_limits<double>::infinity();
        return true;
    }

    int sign = 1;
    std::string body = t;
    if (!body.empty() && (body[0] == '+' || body[0] == '-'))
    {
        if (body[0] == '-')
            sign = -1;
        body = body.substr(1);
    }

    try
    {
        if (body.rfind("0x", 0) == 0 || body.rfind("0X", 0) == 0)
        {
            size_t idx = 0;
            unsigned long long v = std::stoull(body.substr(2), &idx, 16);
            if (idx != body.substr(2).size())
                return false;
            out = (double)v * sign;
            return true;
        }
        if (body.rfind("0b", 0) == 0 || body.rfind("0B", 0) == 0)
        {
            size_t idx = 0;
            unsigned long long v = std::stoull(body.substr(2), &idx, 2);
            if (idx != body.substr(2).size())
                return false;
            out = (double)v * sign;
            return true;
        }
        if (body.rfind("0o", 0) == 0 || body.rfind("0O", 0) == 0)
        {
            size_t idx = 0;
            unsigned long long v = std::stoull(body.substr(2), &idx, 8);
            if (idx != body.substr(2).size())
                return false;
            out = (double)v * sign;
            return true;
        }

        size_t idx = 0;
        double v = std::stod(t, &idx);
        if (idx != t.size())
            return false;
        out = v;
        return true;
    }
    catch (...)
    {
        return false;
    }
}

static bool isLikelyGibberishReply(const std::string &raw)
{
    std::string s = trimCopy(raw);
    if (s.empty())
        return true;

    int letters = 0;
    int digits = 0;
    int spaces = 0;
    int punct = 0;
    int controls = 0;
    int cjk = 0;
    int vowels = 0;
    for (unsigned char ch : s)
    {
        if (ch < 0x20 || ch == 0x7F)
        {
            controls++;
            continue;
        }
        if (std::isspace(ch))
        {
            spaces++;
            continue;
        }
        if (std::isalpha(ch))
        {
            letters++;
            char lo = (char)std::tolower(ch);
            if (lo == 'a' || lo == 'e' || lo == 'i' || lo == 'o' || lo == 'u')
                vowels++;
            continue;
        }
        if (std::isdigit(ch))
        {
            digits++;
            continue;
        }
        if (std::ispunct(ch))
        {
            punct++;
            continue;
        }
        if (ch >= 0x80)
            cjk++;
    }

    if (controls > 0)
        return true;

    int nonSpace = std::max(1, (int)s.size() - spaces);
    double punctRatio = (double)punct / (double)nonSpace;
    if (punctRatio > 0.60)
        return true;

    int core = letters + digits + cjk;
    if (core < std::max(4, (int)s.size() / 10))
        return true;

    std::istringstream iss(s);
    std::string tok;
    int tokenCount = 0;
    int shortTokenCount = 0;
    int oddTokenCount = 0;
    int tokIdLike = 0;
    while (iss >> tok)
    {
        tokenCount++;
        if (tok.size() <= 2)
            shortTokenCount++;
        bool hasNonAlpha = false;
        bool hasDigit = false;
        for (unsigned char c : tok)
        {
            if (!std::isalpha(c))
                hasNonAlpha = true;
            if (std::isdigit(c))
                hasDigit = true;
        }
        if (hasNonAlpha || hasDigit)
            oddTokenCount++;
        if (tok.size() > 3 && tok.rfind("tok", 0) == 0)
        {
            bool allNum = true;
            for (size_t i = 3; i < tok.size(); i++)
            {
                if (!std::isdigit((unsigned char)tok[i]))
                {
                    allNum = false;
                    break;
                }
            }
            if (allNum)
                tokIdLike++;
        }
    }

    if (tokIdLike >= 2)
        return true;

    if (cjk == 0 && tokenCount >= 4)
    {
        double shortRatio = (double)shortTokenCount / (double)std::max(1, tokenCount);
        double oddRatio = (double)oddTokenCount / (double)std::max(1, tokenCount);
        if (shortRatio >= 0.75 && oddRatio >= 0.30)
            return true;
    }

    if (cjk == 0 && letters >= 12)
    {
        double vowelRatio = (double)vowels / (double)letters;
        if (vowelRatio < 0.18)
            return true;
    }

    return false;
}

static bool containsInternalContextMarkers(const std::string &raw)
{
    std::string s = raw;
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c)
                   { return (char)std::tolower(c); });
    static const std::array<const char *, 11> markers = {
        "ctx_v2",
        "ctx_stage1",
        "nodes:",
        "edges:",
        "summary:n=",
        "|act=",
        "|deg=",
        "|seed=",
        "|kw=",
        "[tool_call",
        "[tool_result"};
    for (const char *m : markers)
    {
        if (s.find(m) != std::string::npos)
            return true;
    }
    return false;
}

static bool hasExcessUnknownTokens(const std::string &raw)
{
    std::istringstream iss(raw);
    std::string tok;
    int total = 0;
    int unk = 0;
    while (iss >> tok)
    {
        std::string t;
        t.reserve(tok.size());
        for (unsigned char c : tok)
            t.push_back((char)std::tolower(c));
        if (t == "unk" || t == "<unk>")
            unk++;
        total++;
        if (total >= 80)
            break;
    }
    if (total == 0)
        return false;
    return unk >= 3 && (double)unk / (double)total >= 0.25;
}

static double numberOr(const std::string &s, double fallback)
{
    double v = 0.0;
    if (!parseNumberLikeJs(s, v))
        return fallback;
    if (!std::isfinite(v))
        return fallback;
    if (v == 0.0)
        return fallback;
    return v;
}

static double numberOrNaN(const std::string &s, double fallback)
{
    double v = 0.0;
    if (!parseNumberLikeJs(s, v))
        return fallback;
    if (!std::isfinite(v))
        return fallback;
    return v;
}

static int64_t getRobotsRetrainLimitCap()
{
    int64_t cap = static_cast<int64_t>(std::llround(numberOr(getEnv("AI_ROBOTS_RETRAIN_LIMIT_MAX", "10000"), 10000)));
    if (cap < 1)
        cap = 1;
    if (cap > 1000000)
        cap = 1000000;
    return cap;
}

static int64_t getRobotsRetrainBatchDefault()
{
    int64_t batch = static_cast<int64_t>(std::llround(numberOr(getEnv("AI_ROBOTS_RETRAIN_BATCH", "64"), 64)));
    if (batch < 16)
        batch = 16;
    if (batch > 512)
        batch = 512;
    return batch;
}

static int64_t getRobotsRetrainBatchCharsBudget()
{
    int64_t budget = static_cast<int64_t>(std::llround(numberOr(getEnv("AI_ROBOTS_RETRAIN_BATCH_CHARS", "4000000"), 4000000)));
    if (budget < 200000)
        budget = 200000;
    if (budget > 100000000)
        budget = 100000000;
    return budget;
}

static int64_t getRobotsRetrainDocMaxChars()
{
    int64_t cap = static_cast<int64_t>(std::llround(numberOr(getEnv("AI_ROBOTS_RETRAIN_DOC_MAX_CHARS", "6000"), 6000)));
    if (cap < 512)
        cap = 512;
    if (cap > 200000)
        cap = 200000;
    return cap;
}

static int64_t getRobotsRetrainDocMaxTokens()
{
    int64_t cap = static_cast<int64_t>(std::llround(numberOr(getEnv("AI_ROBOTS_RETRAIN_DOC_MAX_TOKENS", "256"), 256)));
    if (cap < 16)
        cap = 16;
    if (cap > 2000)
        cap = 2000;
    return cap;
}

static void ensureDir(const fs::path &dir)
{
    if (!fs::exists(dir))
        fs::create_directories(dir);
}

static fs::path getProjectRoot()
{
    static fs::path cached;
    static bool inited = false;
    if (inited)
        return cached;
    inited = true;

    std::vector<fs::path> starts;
    if (!g_selfPath.empty())
    {
        try
        {
            starts.push_back(fs::absolute(fs::path(g_selfPath)).parent_path());
        }
        catch (...)
        {
        }
    }
    try
    {
        starts.push_back(fs::current_path());
    }
    catch (...)
    {
    }
    try
    {
        fs::path sourceDir = fs::path(__FILE__).parent_path();
        if (sourceDir.empty())
        {
            sourceDir = fs::current_path();
        }
        starts.push_back(fs::absolute(sourceDir));
    }
    catch (...)
    {
    }

    for (const auto &start : starts)
    {
        fs::path cur = start;
        for (int i = 0; i < 6; i++)
        {
            if (fs::exists(cur / "compile.bat") && fs::exists(cur / "main.cpp"))
            {
                cached = fs::absolute(cur);
                return cached;
            }
            if (fs::exists(cur / "conanfile.txt") && fs::exists(cur / "main.cpp"))
            {
                cached = fs::absolute(cur);
                return cached;
            }
            if (fs::exists(cur / "main.cjs"))
            {
                cached = fs::absolute(cur);
                return cached;
            }
            if (fs::exists(cur / "package.json"))
            {
                cached = fs::absolute(cur);
                return cached;
            }
            if (cur.has_parent_path())
                cur = cur.parent_path();
            else
                break;
        }
    }
    for (const auto &start : starts)
    {
        if (!start.empty())
        {
            cached = start;
            return cached;
        }
    }
    try
    {
        cached = fs::current_path();
    }
    catch (...)
    {
        cached = fs::path(".");
    }
    return cached;
}

static std::map<std::string, std::string> parseArgs(int argc, char **argv)
{
    std::map<std::string, std::string> out;
    for (int i = 1; i < argc; i++)
    {
        std::string item = argv[i];
        if (item.rfind("--", 0) != 0)
            continue;
        auto pos = item.find('=');
        if (pos == std::string::npos)
        {
            out[item.substr(2)] = "true";
        }
        else
        {
            out[item.substr(2, pos - 2)] = item.substr(pos + 1);
        }
    }
    return out;
}

// Config 汇总进程启动所需的运行期配置项。
// 调用方式：由配置加载流程填充后传入各子模块。
// 实现思路：按功能域聚合端口、路径、缓存和并发参数。
// 注意事项：字段默认值用于保障最小可运行配置。
// 注意事项：环境变量/命令行覆盖顺序由外层控制。
// 注意事项：该结构仅承载数据，不包含业务逻辑。
struct Config
{
    fs::path baseDir;
    std::string gatewayHost;
    int portGateway{5080};
    int portStudy{5081};
    int aiCount{7};
    int groupCount{3};
    int groupSize{7};
    int sparkNumAI{7};
    std::string sparkBudget{"default"};
    bool singleProc{false};
    bool groupProc{false};
    int groupProcTimeoutMs{12000};
    bool inferMP{false};
    int inferWorkers{1};
    int redisConnectTimeoutMs{1500};
    std::string redisUrl{"redis://127.0.0.1:6379"};
    std::string redisChannel{"AI-model-workspace"};
    fs::path dbPath;
    int redisCacheDb{1};
    std::string redisCachePrefix{"AI079"};
    fs::path snapshotDir;
    fs::path workerFile;
    int maxWorkers{1};
    fs::path shardCache;
    fs::path lmdbRoot;
    std::size_t lmdbMapSizeBytes{512ull * 1024ull * 1024ull};
    std::string searchEndpoint;
    fs::path robotsDir;
    fs::path lemmaCsv;
    bool lemmaAutoload{false};
    std::size_t lemmaMaxBytes{64ull * 1024ull * 1024ull};
    bool lemmaForce{false};
    int robotsWarmupLimit{200};
    bool robotsAutoload{true};
    bool robotsWarmupShuffle{false};
    int robotsChunkMinWords{2};
    int robotsChunkMaxWords{5};
    int kvmCacheMaxEntries{50000};
    bool learningWarmup{false};
    bool syncStandbyOnBoot{false};
    bool testsAutoload{true};
    fs::path testsDir;
    bool disableBarrier{false};
    bool disableRL{false};
    bool disableADV{false};
    bool disableLearning{false};
    bool disableGnnModule{false};
    bool disableContextModule{false};
    bool usingOllama{true};
    bool ollamaFineTuning{true};
    bool inferenceEnabled{true};
    std::string transformerBackend{"ollama"};
    std::string ollamaModel;
    std::string ollamaBaseUrl{"http://127.0.0.1:11434"};
    int ollamaRequestTimeoutMs{120000};
    std::string llamaCppModel;
    std::string llamaCppBaseUrl{"http://127.0.0.1:8080"};
    int llamaCppRequestTimeoutMs{120000};
    std::string bitnetModel;
    std::string bitnetBaseUrl{"http://127.0.0.1:8090"};
    int bitnetRequestTimeoutMs{120000};
    fs::path ggufModelsDir;
    bool externalBackendStyleSim{true};
    bool transformerBootstrap{false};
    int transformerBootstrapDocs{256};
    std::string logMode{"release"};
    fs::path exportDir;
    bool httpLog{false};
};

static Config loadConfig(int argc, char **argv)
{
    const auto args = parseArgs(argc, argv);
    const auto rootDir = getProjectRoot();
    auto arg = [&](const std::string &k, const std::string &def = "") -> std::string
    {
        auto it = args.find(k);
        if (it != args.end())
            return it->second;
        return def;
    };
    auto argOrEnv = [&](const std::string &argKey, const std::string &env, const std::string &def = "")
    {
        auto it = args.find(argKey);
        if (it != args.end() && !it->second.empty())
            return it->second;
        std::string v = getEnv(env);
        if (!v.empty())
            return v;
        return def;
    };
    auto pathArgOrEnv = [&](const std::string &argKey, const std::string &env, const fs::path &defPath)
    {
        std::string raw = argOrEnv(argKey, env, defPath.string());
        fs::path p = raw.empty() ? defPath : fs::path(raw);
        if (p.is_relative())
            p = rootDir / p;
        return fs::absolute(p);
    };

    Config c;
    c.baseDir = pathArgOrEnv("base-dir", "AI_BASE_DIR", rootDir / "runtime_store");
    c.gatewayHost = argOrEnv("gateway-host", "AI_GATEWAY_HOST", "127.0.0.1");
    c.portGateway = (int)numberOrNaN(argOrEnv("port", "CONTROLLER_PORT", "5080"), 5080);
    c.portStudy = (int)numberOrNaN(argOrEnv("study-port", "AI_STUDY_PORT", "5081"), 5081);
    c.httpLog = boolFrom(argOrEnv("http-log", "AI_HTTP_LOG", "false"), false);

    auto aiCountRaw = argOrEnv("ai-count", "AI_COUNT", getEnv("AI_NUM", "7"));
    auto groupCountRaw = argOrEnv("group-count", "AI_GROUP_COUNT", getEnv("GROUP_COUNT", "3"));
    auto groupSizeRaw = argOrEnv("group-size", "AI_GROUP_SIZE", getEnv("GROUP_SIZE", aiCountRaw));
    auto sparkNumAiRaw = argOrEnv("spark-num-ai", "AI_SPARK_NUM_AI", groupSizeRaw);

    double aiCountNum = numberOr(aiCountRaw, 7);
    double groupCountNum = numberOr(groupCountRaw, 3);
    double groupSizeNum = numberOr(groupSizeRaw, numberOr(aiCountRaw, 7));
    double sparkNumAiNum = numberOr(sparkNumAiRaw, numberOr(groupSizeRaw, numberOr(aiCountRaw, 7)));

    c.aiCount = std::max(3, (int)aiCountNum);
    c.groupCount = std::max(1, (int)groupCountNum);
    c.groupSize = std::max(1, (int)groupSizeNum);
    c.sparkNumAI = std::max(1, (int)sparkNumAiNum);
    c.sparkBudget = argOrEnv("spark-budget", "AI_SPARK_BUDGET", "default");

    c.singleProc = boolFrom(argOrEnv("single-proc", "AI_SINGLE_PROC", "false"), false);
    c.groupProc = c.singleProc ? false : boolFrom(argOrEnv("group-proc", "AI_GROUP_PROC", "false"), false);
    c.groupProcTimeoutMs = std::max(500, (int)numberOr(argOrEnv("group-proc-timeout-ms", "AI_GROUP_PROC_TIMEOUT_MS", "12000"), 12000));

    c.inferMP = c.singleProc ? false : boolFrom(argOrEnv("infer-mp", "AI_INFER_MP", "false"), false);
    int cpuCount = std::max(2u, std::thread::hardware_concurrency());
    int inferWorkersRaw = 0;
    inferWorkersRaw = (int)numberOr(argOrEnv("infer-workers", "AI_INFER_WORKERS", "0"), 0);
    c.inferWorkers = std::max(1, inferWorkersRaw ? inferWorkersRaw : (cpuCount - 1));

    c.redisConnectTimeoutMs = std::max(200, (int)numberOr(argOrEnv("redis-timeout-ms", "AI_REDIS_CONNECT_TIMEOUT_MS", "1500"), 1500));
    c.redisUrl = argOrEnv("redis-url", "REDIS_URL", "redis://127.0.0.1:6379");
    c.redisChannel = argOrEnv("channel", "AI_REDIS_CHANNEL", "AI-model-workspace");
    c.dbPath = pathArgOrEnv("db-path", "AI_DB_PATH", c.baseDir / "ai_store.sqlite");
    c.redisCacheDb = std::max(0, (int)numberOr(argOrEnv("redis-cache-db", "AI_REDIS_CACHE_DB", "1"), 1));
    c.redisCachePrefix = argOrEnv("redis-cache-prefix", "AI_REDIS_CACHE_PREFIX", "AI079");

    c.snapshotDir = pathArgOrEnv("snapshot-dir", "AI_SNAPSHOT_DIR", rootDir / "snapshots");
    c.workerFile = fs::absolute(rootDir / "memeMergeWorker.cjs");
    c.maxWorkers = std::max(1, (int)cpuCount - 1);
    c.shardCache = fs::absolute(rootDir / "shards_cache.json");

    c.lmdbRoot = pathArgOrEnv("lmdb-dir", "LMDB_DIR", rootDir / "lmdb");
    c.lmdbMapSizeBytes = std::max<std::size_t>(64, (std::size_t)numberOr(argOrEnv("lmdb-map-mb", "AI_LMDB_MAP_MB", "512"), 512)) * 1024ull * 1024ull;

    c.searchEndpoint = argOrEnv("search-endpoint", "AI_SEARCH_ENDPOINT", "");
    c.robotsDir = pathArgOrEnv("robots-dir", "AI_ROBOTS_DIR", rootDir / "robots");
    c.lemmaCsv = pathArgOrEnv("lemma-csv", "AI_LEMMA_CSV", rootDir / "lemma.csv");

    c.lemmaAutoload = boolFrom(argOrEnv("lemma-autoload", "AI_LEMMA_AUTOLOAD", "false"), false);
    c.lemmaMaxBytes = std::max<std::size_t>(1, (std::size_t)numberOr(argOrEnv("lemma-max-mb", "AI_LEMMA_MAX_MB", "64"), 64)) * 1024ull * 1024ull;
    c.lemmaForce = boolFrom(argOrEnv("lemma-force", "AI_LEMMA_FORCE", "false"), false);

    c.robotsWarmupLimit = std::max(0, (int)numberOr(argOrEnv("robots-limit", "AI_ROBOTS_LIMIT", "200"), 0));
    c.robotsAutoload = boolFrom(argOrEnv("robots-autoload", "AI_ROBOTS_AUTOLOAD", "true"), true);
    c.robotsWarmupShuffle = boolFrom(argOrEnv("robots-warmup-shuffle", "AI_ROBOTS_WARMUP_SHUFFLE", "false"), false);
    c.robotsChunkMinWords = std::max(1, (int)numberOr(argOrEnv("robots-chunk-min", "AI_ROBOTS_CHUNK_MIN", "3"), 2));
    c.robotsChunkMaxWords = std::max(c.robotsChunkMinWords, (int)numberOr(argOrEnv("robots-chunk-max", "AI_ROBOTS_CHUNK_MAX", "20"), 5));

    c.kvmCacheMaxEntries = std::max(0, (int)numberOr(argOrEnv("kvm-cache-max", "AI_KVM_CACHE_MAX", "50000"), 0));
    c.learningWarmup = boolFrom(argOrEnv("learning-warmup", "AI_LEARNING_WARMUP", "false"), false);
    c.syncStandbyOnBoot = boolFrom(argOrEnv("sync-standby", "AI_SYNC_STANDBY_ON_BOOT", "false"), false);
    c.testsAutoload = boolFrom(argOrEnv("tests-autoload", "AI_TESTS_AUTOLOAD", "true"), true);
    c.testsDir = fs::absolute(rootDir / "tests");

    c.disableBarrier = boolFrom(argOrEnv("disable-memebarrier", "AI_DISABLE_MEMEBARRIER", "false"), false);
    c.disableRL = boolFrom(argOrEnv("disable-rl", "AI_DISABLE_RL", "false"), false);
    c.disableADV = boolFrom(argOrEnv("disable-adv", "AI_DISABLE_ADV", "false"), false);
    c.disableLearning = boolFrom(argOrEnv("disable-learning", "AI_DISABLE_LEARNING", "false"), false);
    c.disableGnnModule = boolFrom(argOrEnv("disable-gnn-module", "AI_DISABLE_GNN_MODULE", "false"), false);
    c.disableContextModule = boolFrom(argOrEnv("disable-context-module", "AI_DISABLE_CONTEXT_MODULE", "false"), false);
    c.usingOllama = boolFrom(argOrEnv("using-ollama", "AI_USING_OLLAMA", "true"), true);
    c.ollamaFineTuning = boolFrom(argOrEnv("ollama-fine-tuning", "AI_OLLAMA_FINE_TUNING", "true"), true);
    c.inferenceEnabled = boolFrom(argOrEnv("inference-enabled", "AI_INFERENCE_ENABLED", "true"), true);
    c.transformerBackend = c.usingOllama ? (c.ollamaFineTuning ? "ollama-fine-tuning" : "ollama") : "native";
    {
        std::string mode = trimCopy(argOrEnv("transformer-mode", "AI_TRANSFORMER_MODE", ""));
        std::transform(mode.begin(), mode.end(), mode.begin(), [](unsigned char ch)
                       { return static_cast<char>(std::tolower(ch)); });
        if (mode == "native")
        {
            c.inferenceEnabled = true;
            c.usingOllama = false;
            c.ollamaFineTuning = false;
            c.transformerBackend = "native";
        }
        else if (mode == "ollama")
        {
            c.inferenceEnabled = true;
            c.usingOllama = true;
            c.ollamaFineTuning = false;
            c.transformerBackend = "ollama";
        }
        else if (mode == "ollama-fine-tuning" || mode == "ollama_fine_tuning")
        {
            c.inferenceEnabled = true;
            c.usingOllama = true;
            c.ollamaFineTuning = true;
            c.transformerBackend = "ollama-fine-tuning";
        }
        else if (mode == "llamacpp" || mode == "llama.cpp" || mode == "llama_cpp")
        {
            c.inferenceEnabled = true;
            c.usingOllama = false;
            c.ollamaFineTuning = false;
            c.transformerBackend = "llamacpp";
        }
        else if (mode == "bitnet")
        {
            c.inferenceEnabled = true;
            c.usingOllama = false;
            c.ollamaFineTuning = false;
            c.transformerBackend = "bitnet";
        }
        else if (mode == "off" || mode == "false" || mode == "disabled")
        {
            c.inferenceEnabled = false;
            c.transformerBackend = "off";
        }
    }
    c.ollamaModel = argOrEnv("ollama-model", "AI_OLLAMA_MODEL", "");
    c.ollamaBaseUrl = argOrEnv("ollama-base-url", "AI_OLLAMA_BASE_URL", "http://127.0.0.1:11434");
    c.ollamaRequestTimeoutMs = std::max(1000, (int)numberOr(argOrEnv("ollama-timeout-ms", "AI_OLLAMA_TIMEOUT_MS", "120000"), 120000));
    c.llamaCppModel = argOrEnv("llamacpp-model", "AI_LLAMACPP_MODEL", "");
    c.llamaCppBaseUrl = argOrEnv("llamacpp-base-url", "AI_LLAMACPP_BASE_URL", "http://127.0.0.1:8080");
    c.llamaCppRequestTimeoutMs = std::max(1000, (int)numberOr(argOrEnv("llamacpp-timeout-ms", "AI_LLAMACPP_TIMEOUT_MS", "120000"), 120000));
    c.bitnetModel = argOrEnv("bitnet-model", "AI_BITNET_MODEL", "");
    c.bitnetBaseUrl = argOrEnv("bitnet-base-url", "AI_BITNET_BASE_URL", "http://127.0.0.1:8090");
    c.bitnetRequestTimeoutMs = std::max(1000, (int)numberOr(argOrEnv("bitnet-timeout-ms", "AI_BITNET_TIMEOUT_MS", "120000"), 120000));
    c.ggufModelsDir = pathArgOrEnv("gguf-models-dir", "AI_GGUF_MODELS_DIR", rootDir / "GGUF_models");
    c.externalBackendStyleSim = boolFrom(argOrEnv("external-style-sim", "AI_EXTERNAL_STYLE_SIM", "true"), true);

    {
        std::error_code ec;
        fs::create_directories(c.ggufModelsDir, ec);

        auto collectGgufFiles = [&](const fs::path &dir)
        {
            std::vector<fs::path> files;
            std::error_code iterEc;
            if (!fs::exists(dir, iterEc) || !fs::is_directory(dir, iterEc))
                return files;
            fs::recursive_directory_iterator it(dir, fs::directory_options::skip_permission_denied, iterEc);
            fs::recursive_directory_iterator end;
            while (!iterEc && it != end)
            {
                const auto &entry = *it;
                std::error_code fileEc;
                if (entry.is_regular_file(fileEc))
                {
                    std::string ext = entry.path().extension().string();
                    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char ch)
                                   { return static_cast<char>(std::tolower(ch)); });
                    if (ext == ".gguf")
                        files.push_back(fs::absolute(entry.path()));
                }
                it.increment(iterEc);
            }
            std::sort(files.begin(), files.end());
            return files;
        };

        auto ensureGgufModelForBackend = [&](const std::string &backend, const std::string &flagName, std::string &modelField)
        {
            auto candidates = collectGgufFiles(c.ggufModelsDir);
            std::string configured = trimCopy(modelField);
            if (configured.empty())
            {
                if (candidates.size() == 1)
                {
                    modelField = candidates.front().string();
                    return;
                }
                if (candidates.empty())
                {
                    throw std::runtime_error("transformer-mode=" + backend + " requires a .gguf model. Put model into " + c.ggufModelsDir.string() + " or pass --" + flagName);
                }
                std::ostringstream oss;
                oss << "transformer-mode=" << backend << " found multiple .gguf models in " << c.ggufModelsDir.string() << ". please specify --" << flagName << ". candidates=";
                for (size_t i = 0; i < candidates.size(); i++)
                {
                    if (i)
                        oss << ",";
                    oss << candidates[i].filename().string();
                }
                throw std::runtime_error(oss.str());
            }

            fs::path modelPath = fs::path(configured);
            if (modelPath.is_relative())
            {
                fs::path underGguf = c.ggufModelsDir / modelPath;
                std::error_code existsEc;
                if (fs::exists(underGguf, existsEc))
                    modelPath = underGguf;
            }
            modelPath = fs::absolute(modelPath);

            std::error_code fileEc;
            if (!fs::exists(modelPath, fileEc) || !fs::is_regular_file(modelPath, fileEc))
                throw std::runtime_error("invalid " + backend + " model path: " + modelPath.string());

            std::string ext = modelPath.extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char ch)
                           { return static_cast<char>(std::tolower(ch)); });
            if (ext != ".gguf")
                throw std::runtime_error(backend + " model must be a .gguf file: " + modelPath.string());

            modelField = modelPath.string();
        };

        if (c.transformerBackend == "llamacpp")
            ensureGgufModelForBackend("llamacpp", "llamacpp-model", c.llamaCppModel);
        else if (c.transformerBackend == "bitnet")
            ensureGgufModelForBackend("bitnet", "bitnet-model", c.bitnetModel);
    }

    c.transformerBootstrap = boolFrom(argOrEnv("transformer-bootstrap", "AI_TRANSFORMER_BOOTSTRAP", "false"), false);
    c.transformerBootstrapDocs = std::max(16, (int)numberOr(argOrEnv("transformer-bootstrap-docs", "AI_TRANSFORMER_BOOTSTRAP_DOCS", "256"), 256));

    c.logMode = argOrEnv("log-mode", "AI_LOG_MODE", "release");
    c.exportDir = pathArgOrEnv("export-dir", "AI_EXPORT_DIR", rootDir / "runtime_store");
    return c;
}

static void applyFrontendEnvFromConfig(const Config &config)
{
    setRuntimeEnvValue("FRONTEND_PORT", std::to_string(config.portStudy));
    setRuntimeEnvValue("AI_API_BASE", std::string("http://") + config.gatewayHost + ":" + std::to_string(config.portGateway));
    std::string gatewayJwtSecret = getEnv("AI_AUTH_JWT_SECRET", getEnv("AUTH_JWT_SECRET", "dev-secret-change-me"));
    if (!gatewayJwtSecret.empty())
        setRuntimeEnvValue("JWT_SECRET", gatewayJwtSecret);
}

// ------------------ Utilities ------------------

static std::string nowIso()
{
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    std::tm tm{};
#ifdef _WIN32
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", &tm);
    std::ostringstream oss;
    oss << buf << '.' << std::setfill('0') << std::setw(3) << ms.count() << 'Z';
    return oss.str();
}

static int64_t nowEpochMs()
{
    return (int64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

static json getLoadAvg()
{
    json out = json::array();
#ifdef _WIN32
    out.push_back(0.0);
    out.push_back(0.0);
    out.push_back(0.0);
#else
    double loads[3] = {0.0, 0.0, 0.0};
    if (getloadavg(loads, 3) != -1)
    {
        out.push_back(loads[0]);
        out.push_back(loads[1]);
        out.push_back(loads[2]);
    }
    else
    {
        out.push_back(0.0);
        out.push_back(0.0);
        out.push_back(0.0);
    }
#endif
    return out;
}

static json getMemoryUsage()
{
    json mem{{"rss", 0}, {"heapTotal", 0}, {"heapUsed", 0}, {"external", 0}, {"arrayBuffers", 0}};
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX pmc{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS *)&pmc, sizeof(pmc)))
    {
        mem["rss"] = (int64_t)pmc.WorkingSetSize;
        mem["heapTotal"] = (int64_t)pmc.PrivateUsage;
        mem["heapUsed"] = (int64_t)pmc.PrivateUsage;
        mem["external"] = 0;
        mem["arrayBuffers"] = 0;
    }
#else
    long pageSize = sysconf(_SC_PAGESIZE);
    std::ifstream statm("/proc/self/statm");
    long size = 0, resident = 0;
    if (statm >> size >> resident)
    {
        int64_t rss = (int64_t)resident * pageSize;
        mem["rss"] = rss;
        mem["heapTotal"] = rss;
        mem["heapUsed"] = rss;
        mem["external"] = 0;
        mem["arrayBuffers"] = 0;
    }
#endif
    return mem;
}

static double getProcessCpuPercent()
{
#ifdef _WIN32
    auto toU64 = [](const FILETIME &ft) -> uint64_t
    {
        ULARGE_INTEGER u{};
        u.LowPart = ft.dwLowDateTime;
        u.HighPart = ft.dwHighDateTime;
        return u.QuadPart;
    };

    FILETIME creation{}, exit{}, kernel{}, user{}, now{};
    if (!GetProcessTimes(GetCurrentProcess(), &creation, &exit, &kernel, &user))
        return 0.0;
    GetSystemTimeAsFileTime(&now);

    static std::mutex cpuMu;
    static uint64_t lastWall = 0;
    static uint64_t lastProc = 0;
    std::lock_guard<std::mutex> lock(cpuMu);

    uint64_t wall = toU64(now);
    uint64_t proc = toU64(kernel) + toU64(user);
    if (lastWall == 0 || wall <= lastWall || proc < lastProc)
    {
        lastWall = wall;
        lastProc = proc;
        return 0.0;
    }
    uint64_t dWall = wall - lastWall;
    uint64_t dProc = proc - lastProc;
    lastWall = wall;
    lastProc = proc;

    const double cpuCount = (double)std::max(1u, std::thread::hardware_concurrency());
    double pct = (double)dProc / ((double)dWall * cpuCount) * 100.0;
    if (!std::isfinite(pct))
        return 0.0;
    return std::max(0.0, std::min(100.0, pct));
#else
    static std::mutex cpuMu;
    static auto lastWall = std::chrono::steady_clock::now();
    static clock_t lastCpu = std::clock();
    std::lock_guard<std::mutex> lock(cpuMu);
    auto now = std::chrono::steady_clock::now();
    clock_t cpuNow = std::clock();
    auto wallMs = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastWall).count();
    if (wallMs <= 0)
        return 0.0;
    double cpuMs = (double)(cpuNow - lastCpu) * 1000.0 / (double)CLOCKS_PER_SEC;
    lastWall = now;
    lastCpu = cpuNow;
    const double cpuCount = (double)std::max(1u, std::thread::hardware_concurrency());
    double pct = cpuMs / ((double)wallMs * cpuCount) * 100.0;
    if (!std::isfinite(pct))
        return 0.0;
    return std::max(0.0, std::min(100.0, pct));
#endif
}

static std::string sha1Hex(const std::string &s)
{
    auto digest = trantor::utils::sha1(s);
    return trantor::utils::toHexString(digest);
}

static std::string randomHex(size_t bytes)
{
    std::random_device rd;
    std::uniform_int_distribution<int> dist(0, 255);
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for (size_t i = 0; i < bytes; i++)
    {
        int v = dist(rd);
        oss << std::setw(2) << (v & 0xff);
    }
    return oss.str();
}

static std::string randomSessionId()
{
    return randomHex(8);
}

static uint32_t hashStrSimple(const std::string &s, uint32_t seed = 1315423911u)
{
    uint32_t hash = seed;
    for (unsigned char c : s)
    {
        hash ^= ((hash << 5) + c + (hash >> 2));
    }
    return hash;
}

static std::string readTextFile(const fs::path &file)
{
    std::ifstream in(file, std::ios::binary);
    if (!in)
        return {};
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

#ifdef _WIN32
using SocketHandle = SOCKET;
static constexpr SocketHandle kInvalidSocket = INVALID_SOCKET;
static void closeSocket(SocketHandle s)
{
    if (s != INVALID_SOCKET)
        closesocket(s);
}
#else
using SocketHandle = int;
static constexpr SocketHandle kInvalidSocket = -1;
static void closeSocket(SocketHandle s)
{
    if (s >= 0)
        close(s);
}
#endif

static bool sendAll(SocketHandle sock, const std::string &data)
{
    const char *buf = data.c_str();
    size_t total = 0;
    while (total < data.size())
    {
#ifdef _WIN32
        int sent = ::send(sock, buf + total, (int)(data.size() - total), 0);
#else
        ssize_t sent = ::send(sock, buf + total, data.size() - total, 0);
#endif
        if (sent <= 0)
            return false;
        total += (size_t)sent;
    }
    return true;
}

// SharedMemorySlice 表示共享内存中的一段偏移与大小信息。
// 调用方式：在跨进程数据交换时作为切片描述符传递。
// 实现思路：记录起始偏移和长度并附带标识字段。
// 注意事项：偏移与长度需由调用方确保边界合法。
// 注意事项：该结构不负责共享内存生命周期管理。
// 注意事项：跨平台使用时需关注地址和对齐约束。
struct SharedMemorySlice
{
    std::shared_ptr<std::vector<uint8_t>> owner;
    const uint8_t *ptr{nullptr};
    size_t bytes{0};

    static SharedMemorySlice fromVector(std::shared_ptr<std::vector<uint8_t>> data)
    {
        SharedMemorySlice s;
        s.owner = std::move(data);
        if (s.owner && !s.owner->empty())
        {
            s.ptr = s.owner->data();
            s.bytes = s.owner->size();
        }
        return s;
    }

    // empty 判断当前切片是否为空。
    // 调用方式：在消费共享数据前先做空值短路。
    // 实现思路：检查指针为空或字节数为零。
    // 注意事项：返回 true 不代表 owner 一定为空。
    bool empty() const { return ptr == nullptr || bytes == 0; }
};

static constexpr int kSharedMemoryProtocolVersion = 2;

static std::string currentOutsidesLinkMode()
{
    return AI_OUTSIDES_LINK_MODE;
}

static json buildShmEnvelope(const json &payload,
                             const std::string &controller,
                             const std::string &backend,
                             const std::string &phase)
{
    return json{{"shmProtocolVersion", kSharedMemoryProtocolVersion},
                {"transport", "inproc-vector-shm2"},
                {"phase", phase},
                {"controller", controller},
                {"backend", backend},
                {"timestampMs", nowEpochMs()},
                {"payload", payload}};
}

static bool unpackShmEnvelope(const json &doc, json &payloadOut)
{
    if (!doc.is_object())
        return false;
    if (!doc.contains("shmProtocolVersion") || !doc.contains("payload"))
        return false;
    int protocol = doc.value("shmProtocolVersion", 1);
    if (protocol <= 0 || protocol > kSharedMemoryProtocolVersion)
        return false;
    payloadOut = doc["payload"];
    return true;
}

// ProcessReservedArena 负责预留进程级连续内存块。
// 调用方式：启动阶段初始化后供高峰路径平滑内存分配抖动。
// 实现思路：按配置大小分配并做触页预热。
// 注意事项：该机制用于稳定性，不改变业务语义。
// 注意事项：失败时应退化而非中断主流程。
// 注意事项：线程安全由内部互斥锁保证。
class ProcessReservedArena
{
public:
    void initialize(size_t bytes)
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (initialized_)
            return;
        bytes = std::max<size_t>(8ull * 1024ull * 1024ull, bytes);
        bytes = std::min<size_t>(2ull * 1024ull * 1024ull * 1024ull, bytes);
        try
        {
            block_.resize(bytes);
            const size_t page = 4096;
            for (size_t i = 0; i < block_.size(); i += page)
                block_[i] = 0;
            reservedBytes_ = block_.size();
            initialized_ = true;
        }
        catch (...)
        {
            block_.clear();
            reservedBytes_ = 0;
            initialized_ = true;
        }
    }

    size_t reservedBytes() const
    {
        return reservedBytes_;
    }

private:
    std::vector<uint8_t> block_;
    size_t reservedBytes_{0};
    bool initialized_{false};
    mutable std::mutex mu_;
};

static ProcessReservedArena &globalReservedArena()
{
    static ProcessReservedArena arena;
    return arena;
}

static void initProcessMemoryReservation()
{
    static std::once_flag once;
    std::call_once(once, []()
                   {
        double mb = numberOr(getEnv("AI_RESERVED_MEM_MB", "256"), 256.0);
        if (!std::isfinite(mb) || mb < 8.0)
            mb = 256.0;
        size_t bytes = (size_t)(mb * 1024.0 * 1024.0);
        globalReservedArena().initialize(bytes); });
}

static bool recvExact(SocketHandle sock, uint8_t *dst, size_t bytes, int timeoutMs)
{
    size_t got = 0;
    while (got < bytes)
    {
        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(sock, &rfds);
        timeval tv;
        tv.tv_sec = timeoutMs / 1000;
        tv.tv_usec = (timeoutMs % 1000) * 1000;
        int rc = select((int)(sock + 1), &rfds, nullptr, nullptr, &tv);
        if (rc <= 0)
            return false;
#ifdef _WIN32
        int n = ::recv(sock, reinterpret_cast<char *>(dst + got), (int)(bytes - got), 0);
#else
        ssize_t n = ::recv(sock, dst + got, bytes - got, 0);
#endif
        if (n <= 0)
            return false;
        got += (size_t)n;
    }
    return true;
}

static bool sendPacket(SocketHandle sock, const std::vector<uint8_t> &payload)
{
    if (payload.size() > (size_t)std::numeric_limits<uint32_t>::max())
        return false;
    uint32_t len = htonl((uint32_t)payload.size());
    std::string header(reinterpret_cast<const char *>(&len), sizeof(len));
    if (!sendAll(sock, header))
        return false;

    size_t sent = 0;
    while (sent < payload.size())
    {
#ifdef _WIN32
        int n = ::send(sock, reinterpret_cast<const char *>(payload.data() + sent), (int)(payload.size() - sent), 0);
#else
        ssize_t n = ::send(sock, payload.data() + sent, payload.size() - sent, 0);
#endif
        if (n <= 0)
            return false;
        sent += (size_t)n;
    }
    return true;
}

static bool recvPacket(SocketHandle sock, std::vector<uint8_t> &payload, int timeoutMs, size_t maxBytes = 64ull * 1024ull * 1024ull)
{
    payload.clear();
    uint32_t lenBe = 0;
    if (!recvExact(sock, reinterpret_cast<uint8_t *>(&lenBe), sizeof(lenBe), timeoutMs))
        return false;
    uint32_t len = ntohl(lenBe);
    if (len == 0)
        return true;
    if ((size_t)len > maxBytes)
        return false;
    payload.resize((size_t)len);
    return recvExact(sock, payload.data(), payload.size(), timeoutMs);
}

static bool recvLine(SocketHandle sock, std::string &out, int timeoutMs)
{
    out.clear();
    char ch;
    while (true)
    {
        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(sock, &rfds);
        timeval tv;
        tv.tv_sec = timeoutMs / 1000;
        tv.tv_usec = (timeoutMs % 1000) * 1000;
        int rc = select((int)(sock + 1), &rfds, nullptr, nullptr, &tv);
        if (rc <= 0)
            return false;
#ifdef _WIN32
        int n = ::recv(sock, &ch, 1, 0);
#else
        ssize_t n = ::recv(sock, &ch, 1, 0);
#endif
        if (n <= 0)
            return false;
        if (ch == '\n')
            break;
        if (ch != '\r')
            out.push_back(ch);
    }
    return true;
}

static std::string urlEncode(const std::string &s)
{
    std::ostringstream oss;
    for (unsigned char c : s)
    {
        if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~')
        {
            oss << c;
        }
        else if (c == ' ')
        {
            oss << '%';
            oss << std::hex << std::uppercase << (int)c << std::nouppercase << std::dec;
        }
        else
        {
            oss << '%';
            oss << std::hex << std::uppercase << (int)c << std::nouppercase << std::dec;
        }
    }
    return oss.str();
}

static std::string urlDecode(const std::string &s)
{
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); i++)
    {
        char c = s[i];
        if (c == '+')
        {
            out.push_back(' ');
            continue;
        }
        if (c == '%' && i + 2 < s.size())
        {
            auto hex = s.substr(i + 1, 2);
            char *end = nullptr;
            long v = std::strtol(hex.c_str(), &end, 16);
            if (end && *end == '\0')
            {
                out.push_back(static_cast<char>(v));
                i += 2;
                continue;
            }
        }
        out.push_back(c);
    }
    return out;
}

// HotMatrixCache 提供矩阵热点缓存与命中统计能力。
// 调用方式：在矩阵计算前后分别调用 get/put 管理热点项。
// 实现思路：本地缓存优先，按策略可同步到外部缓存层。
// 注意事项：缓存命中前需校验维度与版本一致性。
// 注意事项：缓存不可替代源数据正确性校验。
// 注意事项：高并发场景需配合外层限流策略。
class HotMatrixCache
{
public:
    HotMatrixCache(std::string redisUrl, int db, std::string prefix)
        : redisUrl_(std::move(redisUrl)), db_(db), prefix_(std::move(prefix))
    {
#ifdef HAVE_REDIS
        try
        {
            sw::redis::ConnectionOptions opts(redisUrl_);
            opts.db = db_;
            redis_ = std::make_shared<sw::redis::Redis>(opts);
        }
        catch (...)
        {
            redis_.reset();
        }
#endif
    }

    bool enabled() const
    {
#ifdef HAVE_REDIS
        return (bool)redis_;
#else
        return false;
#endif
    }

    bool getMatrix(const std::string &key, int rows, int cols, std::vector<std::vector<float>> &out)
    {
#ifdef HAVE_REDIS
        if (!redis_)
            return false;
        try
        {
            auto val = redis_->get(prefix_ + ":data:" + key);
            if (!val)
                return false;
            json payload = json::parse(*val, nullptr, false);
            if (payload.is_discarded())
                return false;
            int r = payload.value("rows", 0);
            int c = payload.value("cols", 0);
            if (r <= 0 || c <= 0)
                return false;
            if (rows > 0 && rows != r)
                return false;
            if (cols > 0 && cols != c)
                return false;
            auto data = payload.value("data", std::vector<float>{});
            if (data.size() != (size_t)r * (size_t)c)
                return false;
            out.assign((size_t)r, std::vector<float>((size_t)c, 0.0f));
            for (int i = 0; i < r; i++)
            {
                for (int j = 0; j < c; j++)
                {
                    out[(size_t)i][(size_t)j] = data[(size_t)i * (size_t)c + (size_t)j];
                }
            }
            redis_->incr(prefix_ + ":hits:" + key);
            return true;
        }
        catch (...)
        {
            return false;
        }
#else
        (void)key;
        (void)rows;
        (void)cols;
        (void)out;
        return false;
#endif
    }

    void recordAndMaybeSet(const std::string &key,
                           const std::vector<std::vector<float>> &matrix,
                           int ttlSeconds,
                           int hotThreshold = 3)
    {
#ifdef HAVE_REDIS
        if (!redis_)
            return;
        int hits = 0;
        {
            std::lock_guard<std::mutex> lock(mu_);
            hits = ++localHits_[key];
        }
        if (hits < hotThreshold)
            return;
        try
        {
            int rows = (int)matrix.size();
            int cols = rows > 0 ? (int)matrix[0].size() : 0;
            std::vector<float> flat;
            flat.reserve((size_t)rows * (size_t)cols);
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                    flat.push_back(matrix[(size_t)i][(size_t)j]);
            }
            json payload{{"rows", rows}, {"cols", cols}, {"data", flat}};
            std::string k = prefix_ + ":data:" + key;
            redis_->set(k, payload.dump());
            if (ttlSeconds > 0)
                redis_->expire(k, ttlSeconds);
        }
        catch (...)
        {
        }
#else
        (void)key;
        (void)matrix;
        (void)ttlSeconds;
        (void)hotThreshold;
#endif
    }

private:
    std::string redisUrl_;
    int db_{1};
    std::string prefix_;
#ifdef HAVE_REDIS
    std::shared_ptr<sw::redis::Redis> redis_;
#endif
    std::mutex mu_;
    std::unordered_map<std::string, int> localHits_;
};

static json parseFormUrlEncoded(const std::string &body)
{
    json out = json::object();
    size_t start = 0;
    while (start <= body.size())
    {
        size_t amp = body.find('&', start);
        if (amp == std::string::npos)
            amp = body.size();
        std::string pair = body.substr(start, amp - start);
        size_t eq = pair.find('=');
        std::string key = eq == std::string::npos ? pair : pair.substr(0, eq);
        std::string val = eq == std::string::npos ? "" : pair.substr(eq + 1);
        key = urlDecode(key);
        val = urlDecode(val);
        if (!key.empty())
        {
            if (!out.contains(key))
            {
                out[key] = val;
            }
            else if (out[key].is_array())
            {
                out[key].push_back(val);
            }
            else
            {
                json arr = json::array();
                arr.push_back(out[key]);
                arr.push_back(val);
                out[key] = arr;
            }
        }
        if (amp == body.size())
            break;
        start = amp + 1;
    }
    return out;
}

static std::string normalizeUrlSimple(const std::string &url)
{
    if (url.empty())
        return {};
    std::string out = url;
    auto hashPos = out.find('#');
    if (hashPos != std::string::npos)
        out.erase(hashPos);
    auto qPos = out.find('?');
    if (qPos != std::string::npos)
    {
        std::string base = out.substr(0, qPos);
        std::string query = out.substr(qPos + 1);
        std::vector<std::string> parts;
        std::string cur;
        for (char c : query)
        {
            if (c == '&')
            {
                if (!cur.empty())
                    parts.push_back(cur);
                cur.clear();
            }
            else
                cur.push_back(c);
        }
        if (!cur.empty())
            parts.push_back(cur);
        std::vector<std::string> kept;
        for (auto &p : parts)
        {
            auto eq = p.find('=');
            std::string key = (eq == std::string::npos) ? p : p.substr(0, eq);
            std::string lower = key;
            std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
            if (lower.rfind("utm_", 0) == 0 || lower == "fbclid" || lower == "gclid")
                continue;
            kept.push_back(p);
        }
        if (!kept.empty())
        {
            std::string q;
            for (size_t i = 0; i < kept.size(); i++)
            {
                if (i)
                    q += "&";
                q += kept[i];
            }
            out = base + "?" + q;
        }
        else
        {
            out = base;
        }
    }
    return out;
}

static bool sameSiteSimple(const std::string &baseUrl, const std::string &targetUrl)
{
    auto hostOf = [](const std::string &u)
    {
        auto pos = u.find("//");
        if (pos == std::string::npos)
            return std::string();
        auto start = pos + 2;
        auto end = u.find('/', start);
        std::string host = u.substr(start, end == std::string::npos ? u.size() - start : end - start);
        if (host.rfind("www.", 0) == 0)
            host = host.substr(4);
        return host;
    };
    auto h1 = hostOf(baseUrl);
    auto h2 = hostOf(targetUrl);
    return !h1.empty() && h1 == h2;
}

// FetchResult 用于承载一次 HTTP 抓取的完整结果。
// 调用方式：由抓取函数返回给上层解析与清洗流程。
// 实现思路：集中保存正文、状态码、最终 URL 与字节数。
// 注意事项：字段可能为空或为默认值，需由调用方兜底。
// 注意事项：status 不等于 200 时不代表 body 一定为空。
// 注意事项：该结构仅做数据传递，不承担重试策略。
#ifdef HAVE_CURL
struct FetchResult
{
    std::string body;
    std::string contentType;
    long status{0};
    std::string finalUrl;
    size_t bytes{0};
};
#endif

// CurlWriteCtx 描述 libcurl 写回调的输出上下文。
// 调用方式：创建后作为 userdata 传给 CURLOPT_WRITEFUNCTION。
// 实现思路：保存目标字符串指针与最大字节上限。
// 注意事项：out 指针必须在回调生命周期内保持有效。
// 注意事项：maxBytes 为 0 时表示不限制回写长度。
// 注意事项：超过上限会主动截断并通过回调返回中止。
#ifdef HAVE_CURL
struct CurlWriteCtx
{
    std::string *out;
    size_t maxBytes;
};
#endif

#ifdef HAVE_CURL
// curlWriteCb 处理 libcurl 分块下载时的数据写入。
// 调用方式：由 libcurl 在收到网络数据时自动回调。
// 实现思路：按最大容量限制写入并返回已消费字节数。
// 注意事项：返回 0 会触发 curl 提前终止当前下载。
static size_t curlWriteCb(char *ptr, size_t size, size_t nmemb, void *userdata)
{
    auto *ctx = reinterpret_cast<CurlWriteCtx *>(userdata);
    size_t total = size * nmemb;
    if (ctx->maxBytes > 0 && ctx->out->size() + total > ctx->maxBytes)
    {
        size_t allowed = ctx->maxBytes - ctx->out->size();
        if (allowed > 0)
            ctx->out->append(ptr, allowed);
        return 0;
    }
    ctx->out->append(ptr, total);
    return total;
}

// fetchUrl 执行单次 HTTP 获取并返回结构化结果。
// 调用方式：传入 URL、超时、UA 与可选体积限制。
// 实现思路：配置 curl 选项并通过回调累计响应数据。
// 注意事项：网络异常时返回默认结果，调用方需判空校验。
static FetchResult fetchUrl(const std::string &url, int timeoutMs = 10000, const std::string &userAgent = "079ProjectCrawler/1.0", size_t maxBytes = 0)
{
    FetchResult res;
    CURL *curl = curl_easy_init();
    if (!curl)
        return res;
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, timeoutMs);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, userAgent.c_str());
    CurlWriteCtx ctx{&res.body, maxBytes};
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curlWriteCb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &ctx);
    curl_easy_perform(curl);
    char *ct = nullptr;
    curl_easy_getinfo(curl, CURLINFO_CONTENT_TYPE, &ct);
    if (ct)
        res.contentType = ct;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &res.status);
    char *finalUrl = nullptr;
    curl_easy_getinfo(curl, CURLINFO_EFFECTIVE_URL, &finalUrl);
    if (finalUrl)
        res.finalUrl = finalUrl;
    res.bytes = res.body.size();
    curl_easy_cleanup(curl);
    return res;
}
#endif

#ifdef HAVE_GUMBO
static void gumboExtractText(GumboNode *node, std::string &out)
{
    if (!node)
        return;
    if (node->type == GUMBO_NODE_TEXT)
    {
        out.append(node->v.text.text);
        out.push_back(' ');
        return;
    }
    if (node->type != GUMBO_NODE_ELEMENT && node->type != GUMBO_NODE_DOCUMENT)
        return;
    const GumboVector *children = &node->v.element.children;
    for (size_t i = 0; i < children->length; i++)
    {
        gumboExtractText(static_cast<GumboNode *>(children->data[i]), out);
    }
}

static std::string extractTextFromHtml(const std::string &html)
{
    GumboOutput *output = gumbo_parse(html.c_str());
    std::string text;
    gumboExtractText(output->root, text);
    gumbo_destroy_output(&kGumboDefaultOptions, output);
    return text;
}
#endif

#ifdef HAVE_POPPLER
static std::string extractTextFromPdfFile(const fs::path &file)
{
    std::unique_ptr<poppler::document> doc(poppler::document::load_from_file(file.string()));
    if (!doc)
        return {};
    std::string out;
    for (int i = 0; i < doc->pages(); i++)
    {
        auto page = std::unique_ptr<poppler::page>(doc->create_page(i));
        if (!page)
            continue;
        auto bytes = page->text().to_utf8();
        out.append(bytes.begin(), bytes.end());
        out.push_back('\n');
    }
    return out;
}
#endif

static std::vector<std::string> parseJsStringLiterals(const std::string &src)
{
    std::vector<std::string> out;
    const size_t n = src.size();
    for (size_t i = 0; i < n; i++)
    {
        char quote = src[i];
        if (quote != '\'' && quote != '"')
            continue;
        std::string buf;
        i++;
        for (; i < n; i++)
        {
            char c = src[i];
            if (c == '\\' && i + 1 < n)
            {
                char next = src[i + 1];
                if (next == 'n')
                {
                    buf.push_back('\n');
                    i++;
                    continue;
                }
                if (next == 'r')
                {
                    buf.push_back('\r');
                    i++;
                    continue;
                }
                if (next == 't')
                {
                    buf.push_back('\t');
                    i++;
                    continue;
                }
                if (next == '\\' || next == '\'' || next == '"')
                {
                    buf.push_back(next);
                    i++;
                    continue;
                }
                buf.push_back(next);
                i++;
                continue;
            }
            if (c == quote)
                break;
            buf.push_back(c);
        }
        if (!buf.empty())
            out.push_back(buf);
    }
    return out;
}

static bool loadNaturalStopwords(std::unordered_set<std::string> &out)
{
    auto normalize = [](std::string s)
    {
        s = trimCopy(s);
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char ch)
                       { return (char)std::tolower(ch); });
        return s;
    };

    auto collectFromText = [&](const std::string &content)
    {
        size_t before = out.size();
        std::istringstream iss(content);
        std::string line;
        while (std::getline(iss, line))
        {
            line = trimCopy(line);
            if (line.empty())
                continue;
            if (line.rfind("//", 0) == 0 || line.rfind("#", 0) == 0)
                continue;
            for (char sep : {',', ';', '\t', ' '})
                std::replace(line.begin(), line.end(), sep, '\n');
            std::istringstream ps(line);
            std::string token;
            while (std::getline(ps, token))
            {
                token = normalize(token);
                if (token.empty())
                    continue;
                if (token.find_first_of("\r\n") != std::string::npos)
                    continue;
                out.insert(token);
            }
        }
        return out.size() > before;
    };

    auto collectFromJs = [&](const std::string &content)
    {
        size_t before = out.size();
        auto words = parseJsStringLiterals(content);
        for (const auto &w : words)
        {
            std::string token = normalize(w);
            if (token.empty())
                continue;
            if (token.find_first_of(" \t\r\n") != std::string::npos)
                continue;
            out.insert(token);
        }
        return out.size() > before;
    };

    std::vector<fs::path> roots;
    roots.push_back(fs::current_path());
    if (!g_selfPath.empty())
    {
        auto exeDir = fs::absolute(fs::path(g_selfPath)).parent_path();
        roots.push_back(exeDir);
        roots.push_back(exeDir.parent_path());
    }
    roots.push_back(fs::absolute(fs::path(__FILE__).parent_path()));
    roots.push_back(fs::absolute(fs::path(__FILE__).parent_path().parent_path()));

    std::vector<fs::path> candidates;
    for (const auto &root : roots)
    {
        candidates.push_back(root / "node_modules" / "natural" / "lib" / "natural" / "util" / "stopwords.js");
        candidates.push_back(root / "robots" / "stopwords.txt");
        candidates.push_back(root / "doc" / "stopwords.txt");
        candidates.push_back(root / "stopwords.txt");
    }

    std::string envPaths = getEnv("STOPWORDS_PATH", "");
    if (!envPaths.empty())
    {
        for (char &ch : envPaths)
        {
            if (ch == ';' || ch == ',')
                ch = '\n';
        }
        std::istringstream ep(envPaths);
        std::string one;
        while (std::getline(ep, one))
        {
            one = trimCopy(one);
            if (!one.empty())
                candidates.push_back(fs::path(one));
        }
    }

    for (const auto &file : candidates)
    {
        if (!fs::exists(file) || !fs::is_regular_file(file))
            continue;
        std::string content = readTextFile(file);
        if (content.empty())
            continue;
        bool isJs = file.extension() == ".js" || file.extension() == ".mjs" || file.extension() == ".cjs";
        if (isJs)
            collectFromJs(content);
        else
            collectFromText(content);
    }

    if (out.empty())
    {
        static const std::array<const char *, 64> fallback = {
            "a", "an", "the", "and", "or", "but", "if", "then", "else", "for", "to", "of", "in", "on", "at", "by",
            "with", "as", "is", "are", "was", "were", "be", "been", "being", "do", "does", "did", "have", "has", "had", "not",
            "this", "that", "these", "those", "it", "its", "from", "into", "about", "over", "under", "than", "too", "very", "can", "could",
            "will", "would", "should", "i", "you", "he", "she", "we", "they", "them", "my", "your", "our", "their", "的", "了"};
        for (auto *w : fallback)
            out.insert(std::string(w));
    }

    return !out.empty();
}

static const std::unordered_set<std::string> &getStopWords()
{
    static std::unordered_set<std::string> stop;
    static bool loaded = false;
    if (!loaded)
    {
        loaded = true;
        loadNaturalStopwords(stop);
    }
    return stop;
}

static bool isAsciiWordToken(const std::string &token)
{
    if (token.empty())
        return false;
    for (unsigned char c : token)
    {
        if (c >= 0x80)
            return false;
        if (!std::isalnum(c) && c != '_' && c != '-')
            return false;
    }
    return true;
}

static std::vector<std::string> tokenize(const std::string &text)
{
    auto isCjk = [](uint32_t cp)
    {
        return cp >= 0x4e00 && cp <= 0x9fff;
    };
    auto decode = [](const std::string &s, size_t i, uint32_t &cp, size_t &len)
    {
        unsigned char c = (unsigned char)s[i];
        if (c < 0x80)
        {
            cp = c;
            len = 1;
            return true;
        }
        if ((c >> 5) == 0x6 && i + 1 < s.size())
        {
            cp = ((c & 0x1f) << 6) | (s[i + 1] & 0x3f);
            len = 2;
            return true;
        }
        if ((c >> 4) == 0xe && i + 2 < s.size())
        {
            cp = ((c & 0x0f) << 12) | ((s[i + 1] & 0x3f) << 6) | (s[i + 2] & 0x3f);
            len = 3;
            return true;
        }
        if ((c >> 3) == 0x1e && i + 3 < s.size())
        {
            cp = ((c & 0x07) << 18) | ((s[i + 1] & 0x3f) << 12) | ((s[i + 2] & 0x3f) << 6) | (s[i + 3] & 0x3f);
            len = 4;
            return true;
        }
        cp = c;
        len = 1;
        return false;
    };

    std::vector<std::string> parts;
    std::string cur;
    auto flush = [&]()
    {
        if (cur.empty())
            return;
        if (isAsciiWordToken(cur) && getStopWords().count(cur))
        {
            cur.clear();
            return;
        }
        parts.push_back(cur);
        cur.clear();
    };

    for (size_t i = 0; i < text.size();)
    {
        uint32_t cp = 0;
        size_t len = 1;
        decode(text, i, cp, len);
        if (cp == '\r' || cp == '\n' || cp == '\t' || cp == ' ')
        {
            flush();
            i += len;
            continue;
        }
        if (cp < 0x80)
        {
            unsigned char c = (unsigned char)cp;
            if (std::isalnum(c) || c == '_' || c == '-')
            {
                cur.push_back((char)std::tolower(c));
            }
            else
            {
                flush();
            }
            i += len;
            continue;
        }
        if (isCjk(cp))
        {
            cur.append(text.substr(i, len));
        }
        else
        {
            flush();
        }
        i += len;
    }
    flush();
    return parts;
}

static double cosineSim(const std::vector<float> &a, const std::vector<float> &b)
{
    if (a.empty() || b.empty() || a.size() != b.size())
        return 0.0;
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < a.size(); i++)
    {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if (na == 0.0 || nb == 0.0)
        return 0.0;
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

// TorchTextEmbedder 提供轻量文本向量编码能力。
// 调用方式：在需要语义检索或相似度计算时调用。
// 实现思路：构建词表后通过 Embedding 与线性头生成向量。
// 注意事项：首次初始化可能耗时，建议在热身阶段完成。
// 注意事项：词表规模与最大长度会影响内存与精度。
// 注意事项：未就绪状态下调用需由上层进行降级处理。
#ifdef HAVE_TORCH
struct TorchTextEmbedder
{
    bool ready{false};
    int64_t embDim{64};
    int64_t maxLen{40};
    int64_t maxVocab{8000};
    std::unordered_map<std::string, int64_t> stoi;
    std::vector<std::string> itos;
    std::once_flag initFlag;
    torch::nn::Embedding emb{nullptr};
    torch::nn::Linear head{nullptr};

    void buildVocab(const std::unordered_map<std::string, int> &freq)
    {
        std::vector<std::pair<std::string, int>> items(freq.begin(), freq.end());
        std::sort(items.begin(), items.end(), [](auto &a, auto &b)
                  { return a.second > b.second; });
        itos.clear();
        stoi.clear();
        itos.push_back("<pad>");
        itos.push_back("<unk>");
        for (size_t i = 0; i < items.size() && (int64_t)itos.size() < maxVocab; i++)
        {
            stoi[items[i].first] = (int64_t)itos.size();
            itos.push_back(items[i].first);
        }
    }

    std::vector<int64_t> encode(const std::vector<std::string> &tokens) const
    {
        std::vector<int64_t> ids;
        ids.reserve(tokens.size());
        for (const auto &t : tokens)
        {
            auto it = stoi.find(t);
            ids.push_back(it == stoi.end() ? 1 : it->second);
        }
        if (ids.empty())
            ids.push_back(0);
        return ids;
    }

    void ensureInit(const fs::path &robotsDir)
    {
        std::call_once(initFlag, [&]()
                       {
            try {
                std::unordered_map<std::string, int> freq;
                std::vector<std::vector<std::string>> samples;
                size_t files = 0;
                if (fs::exists(robotsDir)) {
                    for (const auto &entry : fs::directory_iterator(robotsDir)) {
                        if (!entry.is_regular_file()) continue;
                        if (entry.path().extension() != ".txt") continue;
                        std::ifstream in(entry.path(), std::ios::binary);
                        if (!in) continue;
                        std::ostringstream ss;
                        ss << in.rdbuf();
                        auto tokens = tokenize(ss.str());
                        if (!tokens.empty()) samples.push_back(tokens);
                        for (const auto &t : tokens) freq[t] += 1;
                        if (++files >= 200) break;
                    }
                }
                if (freq.empty()) return;
                buildVocab(freq);
                int64_t vocabSize = std::max<int64_t>(2, (int64_t)itos.size());
                emb = torch::nn::Embedding(vocabSize, embDim);
                head = torch::nn::Linear(embDim, vocabSize);

                if (!samples.empty()) {
                    torch::optim::Adam opt(emb->parameters(), torch::optim::AdamOptions(1e-3));
                    torch::optim::Adam optHead(head->parameters(), torch::optim::AdamOptions(1e-3));
                    const int epochs = 1;
                    const int batch = 16;
                    for (int e = 0; e < epochs; e++) {
                        for (size_t i = 0; i < samples.size(); i += batch) {
                            size_t end = std::min(samples.size(), i + batch);
                            int64_t bsz = (int64_t)(end - i);
                            auto x = torch::zeros({bsz, maxLen}, torch::kInt64);
                            for (size_t b = 0; b < end - i; b++) {
                                auto ids = encode(samples[i + b]);
                                int64_t len = std::min<int64_t>(maxLen, (int64_t)ids.size());
                                for (int64_t j = 0; j < len; j++) x.index_put_({(int64_t)b, j}, ids[(size_t)j]);
                            }
                            auto input = x.narrow(1, 0, maxLen - 1);
                            auto target = x.narrow(1, 1, maxLen - 1).reshape({-1});
                            opt.zero_grad();
                            optHead.zero_grad();
                            auto ebd = emb(input);
                            auto logits = head(ebd).reshape({-1, vocabSize});
                            auto loss = torch::nn::functional::cross_entropy(logits, target);
                            loss.backward();
                            opt.step();
                            optHead.step();
                        }
                    }
                }
                ready = true;
            } catch (...) {
                ready = false;
            } });
    }

    std::vector<float> embedTokens(const std::vector<std::string> &tokens, int dimOut)
    {
        if (!ready || !emb)
            return {};
        torch::NoGradGuard guard;
        auto ids = encode(tokens);
        auto x = torch::zeros({1, (int64_t)ids.size()}, torch::kInt64);
        for (int64_t j = 0; j < (int64_t)ids.size(); j++)
            x.index_put_({0, j}, ids[(size_t)j]);
        auto ebd = emb(x).mean(1).squeeze(0).contiguous();
        std::vector<float> out(ebd.numel(), 0.0f);
        std::memcpy(out.data(), ebd.data_ptr<float>(), sizeof(float) * out.size());
        if ((int)out.size() == dimOut)
            return out;
        std::vector<float> resized(std::max(1, dimOut), 0.0f);
        int copy = std::min<int>(dimOut, (int)out.size());
        for (int i = 0; i < copy; i++)
            resized[i] = out[(size_t)i];
        return resized;
    }
};

static TorchTextEmbedder &getTorchEmbedder()
{
    static TorchTextEmbedder inst;
    return inst;
}
#endif

// Auto-generated from main.cpp
// Segment: section_tail
// Original lines: 2773-20423

static std::vector<float> textToMiniEmbedding(const std::string &text, int dim = 64)
{
    std::vector<float> vec(std::max(1, dim), 0.0f);
    auto words = tokenize(text);
#ifdef HAVE_TORCH
    if (boolFrom(getEnv("AI_USE_TORCH_EMBED", "false"), false))
    {
        auto &embedder = getTorchEmbedder();
        embedder.ensureInit(getProjectRoot() / "robots");
        if (embedder.ready)
        {
            auto out = embedder.embedTokens(words, dim);
            if (!out.empty())
                return out;
        }
    }
#endif
    for (const auto &w : words)
    {
        auto idx = hashStrSimple(w) % vec.size();
        vec[idx] += 1.0f;
    }
    if (words.size() > 1)
    {
        for (size_t i = 0; i + 1 < words.size(); i++)
        {
            std::string pair = words[i] + "_" + words[i + 1];
            auto idx = hashStrSimple(pair) % vec.size();
            vec[idx] += 0.5f;
        }
    }
    return vec;
}

static std::vector<std::string> perturbTokens(const std::vector<std::string> &words)
{
    if (words.size() <= 1)
        return words;
    size_t idx = (size_t)(std::rand() % words.size());
    std::vector<std::string> out;
    out.reserve(words.size() - 1);
    for (size_t i = 0; i < words.size(); i++)
    {
        if (i != idx)
            out.push_back(words[i]);
    }
    return out;
}

static std::vector<std::string> buildVariants(const std::string &text, int count = 0)
{
    std::vector<std::string> variants;
    if (count <= 0)
        return variants;
    auto words = tokenize(text);
    for (int i = 0; i < count; i++)
    {
        words = perturbTokens(words);
        std::ostringstream oss;
        for (size_t j = 0; j < words.size(); j++)
        {
            if (j)
                oss << " ";
            oss << words[j];
        }
        variants.push_back(oss.str());
        if (words.size() <= 1)
            break;
    }
    return variants;
}

// clamp01 将数值限制在 [0,1] 区间。
// 调用方式：用于概率分数和比率结果归一。
// 实现思路：通过 min/max 双边裁剪实现。
// 注意事项：输入 NaN 会沿标准比较行为传播。
static double clamp01(double x) { return std::max(0.0, std::min(1.0, x)); }
// clamp11 将数值限制在 [-1,1] 区间。
// 调用方式：用于相似度和相关系数截断。
// 实现思路：通过 min/max 双边裁剪实现。
// 注意事项：超界值会被硬截断，可能损失梯度信息。
static double clamp11(double x) { return std::max(-1.0, std::min(1.0, x)); }

// clampInt 从 JSON 中读取整数并执行边界裁剪。
// 调用方式：读取请求参数时传入目标范围与回退值。
// 实现思路：先判断数值类型，再按 min/max 做夹取。
// 注意事项：非数值输入直接返回 fallback。
static int clampInt(const json &value, int minVal, int maxVal, int fallback)
{
    if (!value.is_number())
        return fallback;
    int n = value.get<int>();
    if (n < minVal)
        return minVal;
    if (n > maxVal)
        return maxVal;
    return n;
}

// extractFirstUrl 从文本中提取首个可识别 URL。
// 调用方式：在线检索或链接跟踪前对用户文本调用。
// 实现思路：先匹配 http(s) 链接，再回退匹配 www 形式。
// 注意事项：匹配结果是启发式，调用方应做二次校验。
static std::string extractFirstUrl(const std::string &text)
{
    std::string s = text;
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char c)
                                    { return !std::isspace(c); }));
    if (s.empty())
        return {};
    std::regex r1(R"(https?://[\w\-._~:/?#[\]@!$&'()*+,;=%]+)", std::regex::icase);
    std::smatch m;
    if (std::regex_search(s, m, r1) && m.size() > 0)
        return m[0].str();
    std::regex r2(R"(\b(?:www\.)[a-z0-9\-]+(?:\.[a-z0-9\-]+)+(?:/[\w\-._~:/?#[\]@!$&'()*+,;=%]*)?)", std::regex::icase);
    if (std::regex_search(s, m, r2) && m.size() > 0)
        return std::string("https://") + m[0].str();
    return {};
}

// Rng32 是一个轻量级可复现实用随机数发生器。
// 调用方式：以固定种子初始化后持续调用 next()。
// 实现思路：采用异或移位算法输出 [0,1] 区间随机值。
// 注意事项：该实现用于工程随机，不适用于安全场景。
// 注意事项：相同种子会生成相同序列，便于调试复现。
// 注意事项：多线程共享实例需外部同步。
class Rng32
{
public:
    explicit Rng32(uint32_t seed) : x_(seed ? seed : 0x9e3779b9u) {}
    double next()
    {
        x_ ^= x_ << 13;
        x_ ^= x_ >> 17;
        x_ ^= x_ << 5;
        return (double)(x_ & 0xffffffffu) / (double)0xffffffffu;
    }

private:
    uint32_t x_{0x9e3779b9u};
};

// normalizeBudget 规范化前端预算配置结构。
// 调用方式：传入原始 JSON 并获取可选标准化结果。
// 实现思路：按类型分支做兼容转换与字段清洗。
// 注意事项：非法输入返回空值，调用方应提供默认预算。
static std::optional<json> normalizeBudget(const json &raw)
{
    if (raw.is_null() || raw.is_boolean())
        return std::nullopt;
    if (raw.is_string())
    {
        auto s = raw.get<std::string>();
        std::string lowered;
        lowered.reserve(s.size());
        for (char c : s)
            lowered.push_back((char)std::tolower((unsigned char)c));
        if (lowered == "default" || lowered == "balanced" || lowered == "medium" || lowered == "none")
            return std::nullopt;
        if (lowered == "low" || lowered == "fast")
            return json{{"iteration", 3}, {"reflectionTopMemes", 12}, {"reflectionTopWords", 16}};
        if (lowered == "high" || lowered == "slow" || lowered == "quality")
            return json{{"iteration", 7}, {"reflectionTopMemes", 22}, {"reflectionTopWords", 32}};
        if (!s.empty() && s.front() == '{' && s.back() == '}')
        {
            try
            {
                return normalizeBudget(json::parse(s));
            }
            catch (...)
            {
                return std::nullopt;
            }
        }
        return std::nullopt;
    }
    if (!raw.is_object())
        return std::nullopt;
    json out = json::object();
    auto pickNum = [&](const std::string &key, const std::vector<std::string> &aliases)
    {
        if (raw.contains(key) && raw[key].is_number())
        {
            out[key] = raw[key].get<double>();
            return;
        }
        for (auto &a : aliases)
        {
            if (raw.contains(a) && raw[a].is_number())
            {
                out[key] = raw[a].get<double>();
                return;
            }
        }
    };
    pickNum("mappingDepth", {"depth"});
    pickNum("iteration", {"iters"});
    pickNum("reflectionTopMemes", {"topMemes"});
    pickNum("reflectionTopWords", {"topWords"});
    pickNum("reflectionMinScore", {"minScore"});
    pickNum("radius", {"windowRadius"});
    return out.empty() ? std::nullopt : std::optional<json>(out);
}

static std::optional<json> mergeBudgets(const std::optional<json> &base, const std::optional<json> &override)
{
    if (!base && !override)
        return std::nullopt;
    if (!base)
        return override;
    if (!override)
        return base;
    json out = *base;
    for (auto it = override->begin(); it != override->end(); ++it)
        out[it.key()] = it.value();
    return out;
}

static std::vector<std::string> splitSentences(const std::string &text)
{
    std::string raw = text;
    raw.erase(0, raw.find_first_not_of(" \n\r\t"));
    if (raw.empty())
        return {};
    raw.erase(raw.find_last_not_of(" \n\r\t") + 1);

    std::vector<std::string> rough;
    std::string cur;
    for (size_t i = 0; i < raw.size(); i++)
    {
        char c = raw[i];
        if (c == '\r' || c == '\n')
        {
            auto t = cur;
            t.erase(0, t.find_first_not_of(" \n\r\t"));
            if (!t.empty())
            {
                t.erase(t.find_last_not_of(" \n\r\t") + 1);
                if (!t.empty())
                    rough.push_back(t);
            }
            cur.clear();
            continue;
        }
        cur.push_back(c);
        bool endSentence = (c == '!' || c == '?');
        size_t utf8PunctExtra = 0;
        if (!endSentence)
        {
            if (i + 2 < raw.size())
            {
                unsigned char b1 = static_cast<unsigned char>(raw[i]);
                unsigned char b2 = static_cast<unsigned char>(raw[i + 1]);
                unsigned char b3 = static_cast<unsigned char>(raw[i + 2]);
                bool isCnPeriod = (b1 == 0xE3 && b2 == 0x80 && b3 == 0x82);
                bool isCnExclam = (b1 == 0xEF && b2 == 0xBC && b3 == 0x81);
                bool isCnQuestion = (b1 == 0xEF && b2 == 0xBC && b3 == 0x9F);
                if (isCnPeriod || isCnExclam || isCnQuestion)
                {
                    endSentence = true;
                    utf8PunctExtra = 2;
                    cur.push_back(raw[i + 1]);
                    cur.push_back(raw[i + 2]);
                }
            }
        }
        if (endSentence)
        {
            i += utf8PunctExtra;
            auto t = cur;
            t.erase(0, t.find_first_not_of(" \n\r\t"));
            if (!t.empty())
            {
                t.erase(t.find_last_not_of(" \n\r\t") + 1);
                if (!t.empty())
                    rough.push_back(t);
            }
            cur.clear();
            while (i + 1 < raw.size())
            {
                char n = raw[i + 1];
                if (n == ' ' || n == '\t' || n == '\r' || n == '\n')
                {
                    i++;
                    continue;
                }
                break;
            }
        }
    }
    if (!cur.empty())
    {
        auto t = cur;
        t.erase(0, t.find_first_not_of(" \n\r\t"));
        if (!t.empty())
        {
            t.erase(t.find_last_not_of(" \n\r\t") + 1);
            if (!t.empty())
                rough.push_back(t);
        }
    }

    const int maxWords = 10;
    const int minWords = 2;
    std::vector<std::string> out;

    for (const auto &unit : rough)
    {
        auto tokens = tokenize(unit);
        if ((int)tokens.size() < minWords)
            continue;
        for (size_t i = 0; i < tokens.size(); i += maxWords)
        {
            std::ostringstream oss;
            int count = 0;
            for (size_t j = i; j < tokens.size() && j < i + maxWords; j++)
            {
                if (count++)
                    oss << " ";
                oss << tokens[j];
            }
            if (count < minWords)
            {
                if (!out.empty() && count > 0)
                {
                    out.back() += " " + oss.str();
                }
                continue;
            }
            out.push_back(oss.str());
            if (out.size() >= 12)
                return out;
        }
        if (out.size() >= 12)
            break;
    }
    if (out.size() > 12)
        out.resize(12);
    return out;
}

// SurfacePhrase 表示候选短语及其权重信息。
// 调用方式：作为短语抽取结果元素参与排序与筛选。
// 实现思路：用 phrase 与 weight 两字段描述语义片段。
// 注意事项：weight 仅表示相对强度，不是概率值。
// 注意事项：短语文本应在上游完成基本清洗。
// 注意事项：该结构不负责去重与阈值过滤策略。
struct SurfacePhrase
{
    std::string phrase;
    double weight;
};

// extractSurfacePhrases 从文本抽取候选表层短语。
// 调用方式：输入原文与上限，返回去重后的短语列表。
// 实现思路：综合句子切分、token 与 n-gram 进行采样。
// 注意事项：输出数量受 maxPhrases 与文本长度双重约束。
static std::vector<SurfacePhrase> extractSurfacePhrases(const std::string &text, int maxPhrases = 24)
{
    std::vector<SurfacePhrase> out;
    std::unordered_set<std::string> seen;
    auto push = [&](const std::string &p, double w)
    {
        std::string s = p;
        s.erase(0, s.find_first_not_of(" \n\r\t"));
        if (s.empty())
            return;
        s.erase(s.find_last_not_of(" \n\r\t") + 1);
        if (s.empty())
            return;
        if (s.size() < 2 || s.size() > 160)
            return;
        if (seen.count(s))
            return;
        seen.insert(s);
        out.push_back({s, w});
    };

    for (const auto &s : splitSentences(text))
        push(s, 1.0);

    auto tokens = tokenize(text);
    for (size_t i = 0; i < tokens.size() && (int)out.size() < maxPhrases; i++)
    {
        push(tokens[i], 2.0);
        if ((int)out.size() >= maxPhrases)
            return out;
    }

    int maxN = std::min<int>(5, (int)tokens.size());
    for (int n = 2; n <= maxN; n++)
    {
        for (size_t i = 0; i + n <= tokens.size(); i++)
        {
            std::ostringstream oss;
            for (int j = 0; j < n; j++)
            {
                if (j)
                    oss << " ";
                oss << tokens[i + j];
            }
            double w = (n == 2 ? 3.0 : (n == 3 ? 2.0 : 1.0));
            push(oss.str(), w);
            if ((int)out.size() >= maxPhrases)
                return out;
        }
    }
    return out;
}

// ------------------ Storage ------------------

// LmdbStore 基于 LMDB 实现键值存储后端。
// 调用方式：以库名与根目录初始化后按 KV 接口读写。
// 实现思路：封装环境/事务操作并统一 JSON 编解码。
// 注意事项：依赖底层文件系统权限与路径可写性。
// 注意事项：异常场景会退化为不可用状态而非崩溃。
// 注意事项：跨进程并发访问需遵循 LMDB 限制。
#ifdef HAVE_LMDB
class LmdbStore : public KeyValueStore
{
public:
    LmdbStore(const std::string &name, const fs::path &rootDir, std::size_t mapSizeBytes, bool encodeJSON = true)
        : name_(name), rootDir_(rootDir), mapSizeBytes_(mapSizeBytes), encodeJSON_(encodeJSON)
    {
        ensureDir(rootDir_ / name_);
        fs::path envPath = rootDir_ / name_;
        if (mdb_env_create(&env_) != 0)
        {
            env_ = nullptr;
            return;
        }
        mdb_env_set_maxreaders(env_, 64);
        mdb_env_set_mapsize(env_, mapSizeBytes_);
        if (mdb_env_open(env_, envPath.string().c_str(), 0, 0664) != 0)
        {
            mdb_env_close(env_);
            env_ = nullptr;
            return;
        }
        MDB_txn *txn = nullptr;
        if (mdb_txn_begin(env_, nullptr, 0, &txn) != 0)
            return;
        if (mdb_dbi_open(txn, "default", MDB_CREATE, &dbi_) != 0)
        {
            mdb_txn_abort(txn);
            return;
        }
        mdb_txn_commit(txn);
        ok_ = true;
    }

    ~LmdbStore() override
    {
        if (env_)
        {
            mdb_dbi_close(env_, dbi_);
            mdb_env_close(env_);
        }
    }

    bool ok() const { return ok_; }

    std::optional<json> get(const std::string &key) override
    {
        if (!ok_)
            return std::nullopt;
        MDB_txn *txn = nullptr;
        MDB_val k{key.size(), (void *)key.data()};
        MDB_val v;
        if (mdb_txn_begin(env_, nullptr, MDB_RDONLY, &txn) != 0)
            return std::nullopt;
        int rc = mdb_get(txn, dbi_, &k, &v);
        mdb_txn_abort(txn);
        if (rc != 0)
            return std::nullopt;
        std::string raw((char *)v.mv_data, v.mv_size);
        try
        {
            return encodeJSON_ ? json::parse(raw) : json(raw);
        }
        catch (...)
        {
            return json(raw);
        }
    }

    void put(const std::string &key, const json &value) override
    {
        if (!ok_)
            return;
        MDB_txn *txn = nullptr;
        mdb_txn_begin(env_, nullptr, 0, &txn);
        std::string encoded = encodeJSON_ ? value.dump() : value.get<std::string>();
        MDB_val k{key.size(), (void *)key.data()};
        MDB_val v{encoded.size(), (void *)encoded.data()};
        mdb_put(txn, dbi_, &k, &v, 0);
        mdb_txn_commit(txn);
    }

    void del(const std::string &key) override
    {
        if (!ok_)
            return;
        MDB_txn *txn = nullptr;
        mdb_txn_begin(env_, nullptr, 0, &txn);
        MDB_val k{key.size(), (void *)key.data()};
        mdb_del(txn, dbi_, &k, nullptr);
        mdb_txn_commit(txn);
    }

    std::vector<std::pair<std::string, json>> entries(const std::string &prefix) override
    {
        std::vector<std::pair<std::string, json>> out;
        if (!ok_)
            return out;
        MDB_txn *txn = nullptr;
        MDB_cursor *cursor = nullptr;
        mdb_txn_begin(env_, nullptr, MDB_RDONLY, &txn);
        mdb_cursor_open(txn, dbi_, &cursor);
        MDB_val k, v;
        std::string start = prefix;
        k.mv_size = start.size();
        k.mv_data = (void *)start.data();
        int rc = mdb_cursor_get(cursor, &k, &v, MDB_SET_RANGE);
        while (rc == 0)
        {
            std::string key((char *)k.mv_data, k.mv_size);
            if (!prefix.empty() && key.rfind(prefix, 0) != 0)
                break;
            std::string raw((char *)v.mv_data, v.mv_size);
            try
            {
                out.push_back({key, encodeJSON_ ? json::parse(raw) : json(raw)});
            }
            catch (...)
            {
                out.push_back({key, json(raw)});
            }
            rc = mdb_cursor_get(cursor, &k, &v, MDB_NEXT);
        }
        mdb_cursor_close(cursor);
        mdb_txn_abort(txn);
        return out;
    }

private:
    std::string name_;
    fs::path rootDir_;
    std::size_t mapSizeBytes_;
    bool encodeJSON_{true};
    bool ok_{false};
    MDB_env *env_{nullptr};
    MDB_dbi dbi_{0};
};
#endif
// JsonFileStore 提供基于本地文件的 KV 持久化实现。
// 调用方式：以存储名与目录初始化后按通用接口访问。
// 实现思路：内存缓存 + 后台刷盘线程降低写入抖动。
// 注意事项：进程异常退出可能导致最后批次未落盘。
// 注意事项：大对象频繁更新会带来序列化开销。
// 注意事项：该实现更偏开发环境与轻量部署场景。
class JsonFileStore : public KeyValueStore
{
public:
    JsonFileStore(const std::string &name, const fs::path &rootDir)
        : name_(name), rootDir_(rootDir)
    {
        ensureDir(rootDir_);
        file_ = rootDir_ / (name_ + ".json");
        binFile_ = rootDir_ / (name_ + ".msgpack");
        loadFromDisk();
        startFlushThread();
    }

    ~JsonFileStore() override
    {
        stopFlushThread();
        flush();
    }

    std::optional<json> get(const std::string &key) override
    {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = data_.find(key);
        if (it == data_.end())
            return std::nullopt;
        return it->second;
    }

    void put(const std::string &key, const json &value) override
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (!data_.count(key))
            order_.push_back(key);
        data_[key] = value;
        dirty_ = true;
    }

    void del(const std::string &key) override
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (data_.erase(key) > 0)
        {
            order_.erase(std::remove(order_.begin(), order_.end(), key), order_.end());
        }
        dirty_ = true;
    }

    std::vector<std::pair<std::string, json>> entries(const std::string &prefix) override
    {
        std::vector<std::pair<std::string, json>> out;
        std::lock_guard<std::mutex> lock(mu_);
        for (auto &key : order_)
        {
            auto it = data_.find(key);
            if (it == data_.end())
                continue;
            if (prefix.empty() || key.rfind(prefix, 0) == 0)
                out.push_back(*it);
        }
        return out;
    }

    void flush() override
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (!dirty_)
            return;

        nlohmann::ordered_json doc = nlohmann::ordered_json::object();
        nlohmann::ordered_json items = nlohmann::ordered_json::object();
        nlohmann::ordered_json order = nlohmann::ordered_json::array();
        for (auto &key : order_)
        {
            auto it = data_.find(key);
            if (it != data_.end())
            {
                items[key] = it->second;
                order.push_back(key);
            }
        }
        doc["version"] = 2;
        doc["items"] = std::move(items);
        doc["order"] = std::move(order);

        try
        {
            auto packed = json::to_msgpack(doc);
            fs::path tmpBin = binFile_;
            tmpBin += ".tmp";
            {
                std::ofstream bout(tmpBin, std::ios::binary | std::ios::trunc);
                bout.write(reinterpret_cast<const char *>(packed.data()), (std::streamsize)packed.size());
            }
            if (fs::exists(binFile_))
                fs::remove(binFile_);
            fs::rename(tmpBin, binFile_);
        }
        catch (...)
        {
        }

        try
        {
            fs::path tmpJson = file_;
            tmpJson += ".tmp";
            std::ofstream jout(tmpJson, std::ios::trunc);
            jout << doc.dump(2);
            if (fs::exists(file_))
                fs::remove(file_);
            fs::rename(tmpJson, file_);
        }
        catch (...)
        {
        }

        dirty_ = false;
    }

private:
    bool loadOrderedObject(const nlohmann::ordered_json &j)
    {
        if (!j.is_object())
            return false;
        data_.clear();
        order_.clear();

        auto readLegacy = [&]()
        {
            for (auto it = j.begin(); it != j.end(); ++it)
            {
                json value = it.value();
                if (value.is_string())
                {
                    const auto raw = value.get<std::string>();
                    try
                    {
                        data_[it.key()] = json::parse(raw);
                    }
                    catch (...)
                    {
                        data_[it.key()] = value;
                    }
                }
                else
                {
                    data_[it.key()] = value;
                }
                order_.push_back(it.key());
            }
        };

        if (!j.contains("items") || !j["items"].is_object())
        {
            readLegacy();
            return true;
        }

        const auto &items = j["items"];
        if (j.contains("order") && j["order"].is_array())
        {
            for (const auto &id : j["order"])
            {
                if (!id.is_string())
                    continue;
                std::string key = id.get<std::string>();
                auto it = items.find(key);
                if (it == items.end())
                    continue;
                data_[key] = *it;
                order_.push_back(key);
            }
        }
        for (auto it = items.begin(); it != items.end(); ++it)
        {
            if (!data_.count(it.key()))
            {
                data_[it.key()] = it.value();
                order_.push_back(it.key());
            }
        }
        return true;
    }

    void loadFromDisk()
    {
        if (fs::exists(binFile_))
        {
            try
            {
                std::ifstream in(binFile_, std::ios::binary);
                std::vector<uint8_t> raw((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
                if (!raw.empty())
                {
                    auto j = json::from_msgpack(raw);
                    nlohmann::ordered_json o = j;
                    if (loadOrderedObject(o))
                        return;
                }
            }
            catch (...)
            {
            }
        }

        if (fs::exists(file_))
        {
            try
            {
                std::ifstream in(file_);
                nlohmann::ordered_json j;
                in >> j;
                loadOrderedObject(j);
            }
            catch (...)
            {
            }
        }
    }

    void startFlushThread()
    {
        running_.store(true);
        flushThread_ = std::thread([this]()
                                   {
            while (running_.load()) {
                std::this_thread::sleep_for(std::chrono::seconds(10));
                if (!running_.load()) break;
                flush();
            } });
    }

    void stopFlushThread()
    {
        running_.store(false);
        if (flushThread_.joinable())
            flushThread_.join();
    }

    std::string name_;
    fs::path rootDir_;
    fs::path file_;
    fs::path binFile_;
    bool dirty_{false};
    std::unordered_map<std::string, json> data_;
    std::vector<std::string> order_;
    std::mutex mu_;
    std::atomic<bool> running_{false};
    std::thread flushThread_;
};

// NamespacedStore 在底层存储上叠加命名空间前缀。
// 调用方式：将同一底座按不同业务域隔离键空间。
// 实现思路：统一在 key 前拼接固定前缀后转发调用。
// 注意事项：命名空间为空会抛异常，调用方需预校验。
// 注意事项：前缀策略变更会影响历史数据可见性。
// 注意事项：该封装不改变底层一致性语义。
class NamespacedStore : public KeyValueStore
{
public:
    NamespacedStore(std::shared_ptr<KeyValueStore> base, const std::string &ns)
        : base_(std::move(base))
    {
        std::string trimmed = ns;
        trimmed.erase(trimmed.begin(), std::find_if(trimmed.begin(), trimmed.end(), [](unsigned char ch)
                                                    { return !std::isspace(ch); }));
        trimmed.erase(std::find_if(trimmed.rbegin(), trimmed.rend(), [](unsigned char ch)
                                   { return !std::isspace(ch); })
                          .base(),
                      trimmed.end());
        if (trimmed.empty())
            throw std::runtime_error("NamespacedStore requires a namespace");
        prefix_ = "ns:" + trimmed + ":";
    }

    std::optional<json> get(const std::string &key) override { return base_->get(prefix_ + key); }
    void put(const std::string &key, const json &value) override { base_->put(prefix_ + key, value); }
    void del(const std::string &key) override { base_->del(prefix_ + key); }
    std::vector<std::pair<std::string, json>> entries(const std::string &prefix) override
    {
        auto all = base_->entries(prefix_ + prefix);
        std::vector<std::pair<std::string, json>> out;
        for (auto &kv : all)
        {
            if (kv.first.rfind(prefix_, 0) == 0)
            {
                out.push_back({kv.first.substr(prefix_.size()), kv.second});
            }
        }
        return out;
    }

private:
    std::shared_ptr<KeyValueStore> base_;
    std::string prefix_;
};

// ------------------ Core model ------------------

class RuntimeState;
struct RobotsDocument;

// RuntimeBase 定义运行时控制器的基础能力集合。
// 调用方式：具体运行时实现继承该基类并重写核心接口。
// 实现思路：抽象公共依赖、配置与通用工具函数。
// 注意事项：析构需保证派生类资源正确释放。
// 注意事项：并发成员访问需遵守线程模型约束。
// 注意事项：该类本身不直接承载完整业务流程。
class RuntimeBase
{
public:
    virtual ~RuntimeBase() = default;
    virtual json processInput(const json &payload) = 0;
    virtual json cloneParams() const = 0;
    virtual void setParams(const json &patch) = 0;
    virtual json toSnapshot() = 0;
    virtual void fromSnapshot(const json &snapshot) = 0;
    virtual json ingestDocument(const json &doc) = 0;
    virtual json forgetMemes(const json &payload) = 0;
    virtual json onlineLookup(const std::string &query, const json &options) = 0;
    virtual json onlineLookup(const json &input, const json &options) = 0;
    virtual json getSearchConfig() const = 0;
    virtual json setSearchConfig(const json &patch) = 0;
    virtual std::vector<RobotsDocument> collectRobotsDocuments(const json &options) const = 0;
    virtual std::vector<std::string> listRobotsFiles() const = 0;
    virtual fs::path exportGraphToFile(const json &options) = 0;
    virtual json learnFromDialog(const json &payload, const json &result) = 0;
    virtual std::unordered_map<std::string, double> mapWordsToMemes(const std::vector<std::string> &words) = 0;
    virtual std::vector<std::string> getMemeWords(const std::string &memeId) = 0;
    virtual void graphLink(const std::string &fromId, const std::string &toId, double weight, int direction) = 0;
    virtual const Config &config() const = 0;
};

// SearchConfig 描述在线检索流程的策略与阈值参数。
// 调用方式：由 API 或配置中心下发后驱动检索行为。
// 实现思路：集中管理超时、数量限制与过滤开关。
// 注意事项：参数组合会影响时延、召回和成本。
// 注意事项：变更后建议结合监控指标回归验证。
// 注意事项：默认值用于兼容历史请求行为。
struct SearchConfig
{
    bool enabled{false};
    std::vector<std::string> endpoints;
    std::string active;

    json toJson() const
    {
        return json{{"enabled", enabled}, {"endpoints", endpoints}, {"active", active}};
    }
};

// ModelDefaults 记录模型运行的默认超参数集合。
// 调用方式：当请求未提供参数时作为回退配置。
// 实现思路：聚合采样、长度和权重等常用参数。
// 注意事项：默认值需与当前模型能力匹配。
// 注意事项：外部显式参数通常优先于该结构。
// 注意事项：调整需评估兼容性与稳定性影响。
struct ModelDefaults
{
    double decayFactor{0.5};
    int maxMemeWords{100};
    int minOverlapThreshold{2};
    int memeNgramMin{3};
    int memeNgramMax{14};
    double maliciousThreshold{0.7};
    int learningIterations{3};
    int iteration{5};
    double threshold{3};
    double decay{1};
    double decayK{1};
    int maxLen{16};
    double edgeWeight{1};
    std::string activationType{"relu"};
    std::string transferType{"linear"};
    std::string activationCustom;
    std::string transferCustom;
    int mappingDepth{1};
    int reflectionTopMemes{18};
    int reflectionTopWords{24};
    double reflectionMinScore{1e-6};
};

static ModelDefaults MODEL_DEFAULTS;

// OnlineResearcher 提供在线资料检索与摘要辅助能力。
// 调用方式：在复杂问题处理中由运行时按需调用。
// 实现思路：封装抓取、过滤、聚合和提示增强流程。
// 注意事项：外部网络失败时应支持降级路径。
// 注意事项：结果需接受安全和可信度约束。
// 注意事项：该组件不应阻塞核心请求主链路。
class OnlineResearcher
{
public:
    OnlineResearcher(RuntimeState *runtime, const std::string &endpoint, int cooldownMs = 5 * 60 * 1000, int cacheSize = 128)
        : runtime_(runtime), endpoint_(endpoint), cooldownMs_(cooldownMs), cacheSize_(cacheSize) {}

    // setEndpoint 更新在线检索服务地址。
    // 调用方式：配置热更新后调用以切换后端。
    // 实现思路：直接替换内部 endpoint 字段。
    // 注意事项：新地址需由调用方保证可达与合法。
    void setEndpoint(const std::string &endpoint) { endpoint_ = endpoint; }
    // setEnabled 开启或关闭在线检索功能。
    // 调用方式：运行期开关控制时调用。
    // 实现思路：更新布尔标记供查询路径判断。
    // 注意事项：关闭后将走本地降级路径。
    void setEnabled(bool enabled) { enabled_ = enabled; }

    json lookup(const json &input, const json &options);

private:
    struct CacheEntry
    {
        int64_t ts{0};
        json payload;
    };
    struct DocEntry
    {
        std::string url;
        std::string title;
        std::string text;
        std::vector<std::string> tokens;
        std::vector<float> embedding;
        std::vector<std::string> aspects;
    };

    RuntimeState *runtime_{nullptr};
    std::string endpoint_;
    int cooldownMs_{5 * 60 * 1000};
    int cacheSize_{128};
    bool enabled_{true};
    std::unordered_map<std::string, CacheEntry> cache_;
    std::unordered_map<std::string, DocEntry> docs_;
    std::unordered_map<std::string, std::unordered_set<std::string>> invertedIndex_;
    std::unordered_map<std::string, std::unordered_set<std::string>> aspectToUrls_;
    std::unordered_map<std::string, std::string> urlToContent_;
    std::mutex indexMu_;

    std::string normalizeKey(const std::vector<std::string> &words) const;
    void pruneCache();
    void remember(const std::string &key, const json &payload);
    std::optional<json> fromCache(const std::string &key);
    json fallback(const std::vector<std::string> &words);
    json crawl(const std::string &startUrl, const json &crawlReq, const json &options);
    std::vector<std::string> loadSeedUrls(int maxSeeds) const;
    std::vector<std::string> extractLinksFromHtml(const std::string &html, const std::string &baseUrl) const;
    std::string cleanTextAdvanced(const std::string &text) const;
    std::vector<std::string> pickAspects(const std::vector<std::string> &tokens, int maxAspects) const;
    void indexDocument(const DocEntry &doc);
    json queryLocalIndex(const std::vector<std::string> &words, int limitUrls, int limitAspects, int maxClusters);
    static std::vector<std::vector<float>> kMeans(const std::vector<std::vector<float>> &vectors, int k, int iters, std::vector<int> &assignments);
};
// relu 计算 ReLU 激活函数。
// 调用方式：在图传播或 MLP 步骤中作为激活。
// 实现思路：负值归零，正值保持原样。
// 注意事项：在 0 点不可导，数值优化需留意。
static double relu(double x) { return x > 0 ? x : 0; }
// leaky_relu 计算 LeakyReLU 激活函数。
// 调用方式：用于避免 ReLU 死亡神经元问题。
// 实现思路：负半轴保留小斜率 0.01。
// 注意事项：斜率固定，若需可配置需在上层扩展。
static double leaky_relu(double x) { return x > 0 ? x : 0.01 * x; }
// sigmoid 计算 Sigmoid 映射。
// 调用方式：用于概率化输出或门控计算。
// 实现思路：按标准 1/(1+e^-x) 公式实现。
// 注意事项：大绝对值输入可能导致梯度饱和。
static double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
static constexpr double kPi = 3.14159265358979323846;
// gelu 计算 GELU 激活近似。
// 调用方式：在更平滑激活需求下替代 ReLU。
// 实现思路：采用 tanh 近似表达式。
// 注意事项：相比 ReLU 计算开销更高。
static double gelu(double x) { return 0.5 * x * (1 + std::tanh(std::sqrt(2.0 / kPi) * (x + 0.044715 * std::pow(x, 3)))); }
// identity_fn 返回输入值本身。
// 调用方式：在需要关闭激活时作为透传函数。
// 实现思路：直接返回参数不做变换。
// 注意事项：上游需自行保证数值稳定性。
static double identity_fn(double x) { return x; }

// ExprToken 表示表达式解析阶段的最小语法单元。
// 调用方式：tokenizeExpr/toRpn/evalRpn 在词法与语法阶段共享。
// 实现思路：统一描述类型、文本值、优先级和函数参数信息。
// 注意事项：同一 token 的字段有效性由 type 决定。
// 注意事项：Op/Func 类型会依赖 precedence/arity 字段。
// 注意事项：该结构仅承载解析信息，不直接执行计算。
struct ExprToken
{
    enum class Type
    {
        Number,
        Var,
        Op,
        Func,
        LParen,
        RParen,
        Comma
    } type{Type::Number};
    std::string text;
    double value{0.0};
    int precedence{0};
    bool rightAssoc{false};
    int arity{0};
};

// 判断字符是否可作为标识符组成部分。
// 调用方式：表达式词法分析时识别变量名与函数名。
// 实现思路：允许字母数字、下划线与点号。
// 注意事项：仅覆盖 ASCII 规则，不做完整 Unicode 标识符支持。
static bool isIdentChar(char c)
{
    return std::isalnum((unsigned char)c) || c == '_' || c == '.';
}

// 返回内置函数的参数个数。
// 调用方式：RPN 构建与求值阶段校验函数调用时使用。
// 实现思路：对特殊函数（min/max/pow）返回 2，其余默认 1。
// 注意事项：新增函数时需要同步更新该映射。
static int funcArity(const std::string &name)
{
    if (name == "min" || name == "max" || name == "pow")
        return 2;
    return 1;
}

// 判断标识符是否为支持的函数名。
// 调用方式：词法分析遇到标识符时区分变量与函数。
// 实现思路：在静态哈希集合中做 O(1) 查询。
// 注意事项：名称比较区分大小写，输入前需自行规范化。
static bool isFuncName(const std::string &name)
{
    static const std::unordered_set<std::string> funcs = {
        "sin", "cos", "tan", "tanh", "exp", "log", "sqrt", "abs", "min", "max", "pow"};
    return funcs.count(name) > 0;
}

// 将表达式字符串切分为词法 token 序列。
// 调用方式：compileExpression 的第一阶段。
// 实现思路：按数字、标识符、运算符和括号逐字符扫描。
// 注意事项：解析失败会设置 ok=false 并返回空/部分结果。
static std::vector<ExprToken> tokenizeExpr(const std::string &expr, bool &ok)
{
    ok = true;
    std::vector<ExprToken> out;
    for (size_t i = 0; i < expr.size();)
    {
        char c = expr[i];
        if (std::isspace((unsigned char)c))
        {
            i++;
            continue;
        }
        if (std::isdigit((unsigned char)c) || c == '.')
        {
            size_t j = i + 1;
            while (j < expr.size() && (std::isdigit((unsigned char)expr[j]) || expr[j] == '.' || expr[j] == 'e' || expr[j] == 'E' || expr[j] == '+' || expr[j] == '-'))
            {
                if ((expr[j] == '+' || expr[j] == '-') && !(expr[j - 1] == 'e' || expr[j - 1] == 'E'))
                    break;
                j++;
            }
            double v = 0.0;
            try
            {
                v = std::stod(expr.substr(i, j - i));
            }
            catch (...)
            {
                ok = false;
                return {};
            }
            out.push_back(ExprToken{ExprToken::Type::Number, "", v});
            i = j;
            continue;
        }
        if (std::isalpha((unsigned char)c) || c == '_')
        {
            size_t j = i + 1;
            while (j < expr.size() && isIdentChar(expr[j]))
                j++;
            std::string name = expr.substr(i, j - i);
            if (name == "x")
            {
                out.push_back(ExprToken{ExprToken::Type::Var, name, 0.0});
            }
            else if (name == "pi")
            {
                out.push_back(ExprToken{ExprToken::Type::Number, "", kPi});
            }
            else if (name == "e")
            {
                out.push_back(ExprToken{ExprToken::Type::Number, "", std::exp(1.0)});
            }
            else if (isFuncName(name))
            {
                out.push_back(ExprToken{ExprToken::Type::Func, name, 0.0, 0, false, funcArity(name)});
            }
            else
            {
                ok = false;
                return {};
            }
            i = j;
            continue;
        }
        if (c == '(')
        {
            out.push_back(ExprToken{ExprToken::Type::LParen});
            i++;
            continue;
        }
        if (c == ')')
        {
            out.push_back(ExprToken{ExprToken::Type::RParen});
            i++;
            continue;
        }
        if (c == ',')
        {
            out.push_back(ExprToken{ExprToken::Type::Comma});
            i++;
            continue;
        }
        if (c == '+' || c == '-' || c == '*' || c == '/' || c == '^')
        {
            ExprToken tok;
            tok.type = ExprToken::Type::Op;
            tok.text = std::string(1, c);
            if (c == '^')
            {
                tok.precedence = 3;
                tok.rightAssoc = true;
            }
            else if (c == '*' || c == '/')
            {
                tok.precedence = 2;
            }
            else
            {
                tok.precedence = 1;
            }
            out.push_back(tok);
            i++;
            continue;
        }
        ok = false;
        return {};
    }
    return out;
}

static std::vector<ExprToken> toRpn(const std::vector<ExprToken> &tokens, bool &ok)
{
    ok = true;
    std::vector<ExprToken> output;
    std::vector<ExprToken> ops;
    ExprToken prev;
    bool hasPrev = false;
    for (size_t i = 0; i < tokens.size(); i++)
    {
        auto tok = tokens[i];
        if (tok.type == ExprToken::Type::Op && tok.text == "-")
        {
            if (!hasPrev || prev.type == ExprToken::Type::Op || prev.type == ExprToken::Type::LParen || prev.type == ExprToken::Type::Comma)
            {
                tok.text = "neg";
                tok.precedence = 4;
                tok.rightAssoc = true;
                tok.arity = 1;
            }
        }
        if (tok.type == ExprToken::Type::Number || tok.type == ExprToken::Type::Var)
        {
            output.push_back(tok);
        }
        else if (tok.type == ExprToken::Type::Func)
        {
            ops.push_back(tok);
        }
        else if (tok.type == ExprToken::Type::Comma)
        {
            while (!ops.empty() && ops.back().type != ExprToken::Type::LParen)
            {
                output.push_back(ops.back());
                ops.pop_back();
            }
            if (ops.empty())
            {
                ok = false;
                return {};
            }
        }
        else if (tok.type == ExprToken::Type::Op)
        {
            while (!ops.empty())
            {
                auto top = ops.back();
                if (top.type == ExprToken::Type::Op)
                {
                    if ((tok.rightAssoc && tok.precedence < top.precedence) || (!tok.rightAssoc && tok.precedence <= top.precedence))
                    {
                        output.push_back(top);
                        ops.pop_back();
                        continue;
                    }
                }
                else if (top.type == ExprToken::Type::Func)
                {
                    output.push_back(top);
                    ops.pop_back();
                    continue;
                }
                break;
            }
            ops.push_back(tok);
        }
        else if (tok.type == ExprToken::Type::LParen)
        {
            ops.push_back(tok);
        }
        else if (tok.type == ExprToken::Type::RParen)
        {
            bool matched = false;
            while (!ops.empty())
            {
                auto top = ops.back();
                ops.pop_back();
                if (top.type == ExprToken::Type::LParen)
                {
                    matched = true;
                    break;
                }
                output.push_back(top);
            }
            if (!matched)
            {
                ok = false;
                return {};
            }
            if (!ops.empty() && ops.back().type == ExprToken::Type::Func)
            {
                output.push_back(ops.back());
                ops.pop_back();
            }
        }
        prev = tok;
        hasPrev = true;
    }
    while (!ops.empty())
    {
        if (ops.back().type == ExprToken::Type::LParen || ops.back().type == ExprToken::Type::RParen)
        {
            ok = false;
            return {};
        }
        output.push_back(ops.back());
        ops.pop_back();
    }
    return output;
}

static double evalRpn(const std::vector<ExprToken> &rpn, double x, bool &ok)
{
    ok = true;
    std::vector<double> st;
    st.reserve(rpn.size());
    for (const auto &tok : rpn)
    {
        if (tok.type == ExprToken::Type::Number)
        {
            st.push_back(tok.value);
        }
        else if (tok.type == ExprToken::Type::Var)
        {
            st.push_back(x);
        }
        else if (tok.type == ExprToken::Type::Op)
        {
            if (tok.text == "neg")
            {
                if (st.empty())
                {
                    ok = false;
                    return 0.0;
                }
                double v = st.back();
                st.pop_back();
                st.push_back(-v);
            }
            else
            {
                if (st.size() < 2)
                {
                    ok = false;
                    return 0.0;
                }
                double b = st.back();
                st.pop_back();
                double a = st.back();
                st.pop_back();
                if (tok.text == "+")
                    st.push_back(a + b);
                else if (tok.text == "-")
                    st.push_back(a - b);
                else if (tok.text == "*")
                    st.push_back(a * b);
                else if (tok.text == "/")
                    st.push_back(b == 0.0 ? 0.0 : a / b);
                else if (tok.text == "^")
                    st.push_back(std::pow(a, b));
                else
                {
                    ok = false;
                    return 0.0;
                }
            }
        }
        else if (tok.type == ExprToken::Type::Func)
        {
            int arity = tok.arity ? tok.arity : funcArity(tok.text);
            if ((int)st.size() < arity)
            {
                ok = false;
                return 0.0;
            }
            double b = 0.0, a = 0.0;
            if (arity >= 2)
            {
                b = st.back();
                st.pop_back();
                a = st.back();
                st.pop_back();
            }
            else
            {
                a = st.back();
                st.pop_back();
            }
            double v = 0.0;
            if (tok.text == "sin")
                v = std::sin(a);
            else if (tok.text == "cos")
                v = std::cos(a);
            else if (tok.text == "tan")
                v = std::tan(a);
            else if (tok.text == "tanh")
                v = std::tanh(a);
            else if (tok.text == "exp")
                v = std::exp(a);
            else if (tok.text == "log")
                v = a <= 0 ? 0.0 : std::log(a);
            else if (tok.text == "sqrt")
                v = a < 0 ? 0.0 : std::sqrt(a);
            else if (tok.text == "abs")
                v = std::abs(a);
            else if (tok.text == "min")
                v = std::min(a, b);
            else if (tok.text == "max")
                v = std::max(a, b);
            else if (tok.text == "pow")
                v = std::pow(a, b);
            else
            {
                ok = false;
                return 0.0;
            }
            st.push_back(v);
        }
    }
    if (st.size() != 1 || !std::isfinite(st.back()))
    {
        ok = false;
        return 0.0;
    }
    return st.back();
}

static std::shared_ptr<std::vector<ExprToken>> compileExpression(const std::string &expr)
{
    static std::mutex mu;
    static std::unordered_map<std::string, std::shared_ptr<std::vector<ExprToken>>> cache;
    if (expr.empty())
        return nullptr;
    {
        std::lock_guard<std::mutex> lock(mu);
        auto it = cache.find(expr);
        if (it != cache.end())
            return it->second;
    }
    bool ok = true;
    auto tokens = tokenizeExpr(expr, ok);
    if (!ok)
        return nullptr;
    auto rpn = toRpn(tokens, ok);
    if (!ok)
        return nullptr;
    auto ptr = std::make_shared<std::vector<ExprToken>>(std::move(rpn));
    {
        std::lock_guard<std::mutex> lock(mu);
        cache[expr] = ptr;
    }
    return ptr;
}

static std::function<double(double)> activationFn(const ModelDefaults &params)
{
    if (params.activationType == "identity" || params.activationType == "linear")
        return identity_fn;
    if (params.activationType == "tanh")
        return [](double x)
        { return std::tanh(x); };
    if (params.activationType == "sigmoid")
        return sigmoid;
    if (params.activationType == "leaky_relu")
        return leaky_relu;
    if (params.activationType == "gelu")
        return gelu;
    if (params.activationType == "elu")
        return [](double x)
        { return x >= 0 ? x : (std::exp(x) - 1); };
    if (params.activationType == "softplus")
        return [](double x)
        { return std::log(1 + std::exp(x)); };
    if (params.activationType == "custom" && !params.activationCustom.empty())
    {
        auto rpn = compileExpression(params.activationCustom);
        if (rpn)
        {
            return [rpn](double x)
            {
                bool ok = true;
                double v = evalRpn(*rpn, x, ok);
                return ok ? v : 0.0;
            };
        }
    }
    return relu;
}

struct CSRMatrix
{
    std::vector<uint32_t> rowPtr;
    std::vector<uint32_t> colIdx;
    std::vector<float> values;
    uint32_t nRows{0};
    uint32_t nCols{0};
};

// TensorEngine 提供张量运算与特征处理核心能力。
// 调用方式：由上层学习模块调用其矩阵/向量计算接口。
// 实现思路：封装常用线代、归一化与统计计算流程。
// 注意事项：输入维度不一致时应尽早回退或报错。
// 注意事项：性能路径应减少临时对象分配。
// 注意事项：并发使用需遵守实例线程安全约束。
class TensorEngine
// TensorEngine 提供张量运算与特征处理核心能力。
// 调用方式：由上层学习模块调用其矩阵/向量计算接口。
// 实现思路：封装常用线代、归一化与统计计算流程。
// 注意事项：输入维度不一致时应尽早回退或报错。
// 注意事项：性能路径应减少临时对象分配。
// 注意事项：并发使用需遵守实例线程安全约束。
{
public:
    std::vector<float> spmm(const CSRMatrix &csr, const std::vector<float> &x)
    {
        auto &logger = LoggerCXX::instance();
        if (logger.enabled())
            logger.log(LoggerCXX::Type::COMPUTE,
                       std::string("graph spmm rows=") + std::to_string(csr.nRows) +
                           " cols=" + std::to_string(csr.nCols) +
                           " nnz=" + std::to_string(csr.values.size()));
        std::vector<float> out(csr.nRows, 0.0f);
        for (uint32_t row = 0; row < csr.nRows; row++)
        {
            float acc = 0;
            for (uint32_t idx = csr.rowPtr[row]; idx < csr.rowPtr[row + 1]; idx++)
            {
                uint32_t col = csr.colIdx[idx];
                float w = csr.values[idx];
                float xv = (col < x.size()) ? x[col] : 0;
                if (!std::isfinite(w) || !std::isfinite(xv))
                    continue;
                acc += w * xv;
            }
            out[row] = std::isfinite(acc) ? acc : 0;
        }
        return out;
    }

    std::vector<float> iteratePropagation(const CSRMatrix &csr, const std::vector<float> &seeds, int steps, const std::function<double(double)> &actFn, double decayK, double damp = 0.02)
    {
        auto &logger = LoggerCXX::instance();
        int64_t startMs = nowEpochMs();
        std::vector<float> state = seeds;
        std::vector<float> next(seeds.size());
        for (int s = 0; s < steps; s++)
        {
            next = spmm(csr, state);
            for (size_t i = 0; i < next.size(); i++)
            {
                double si = std::isfinite(state[i]) ? state[i] : 0;
                double ni = std::isfinite(next[i]) ? next[i] : 0;
                double raw = si + ni - (decayK * si * damp);
                double y = 0;
                try
                {
                    y = actFn(raw);
                }
                catch (...)
                {
                    y = 0;
                }
                state[i] = std::isfinite(y) ? (float)y : 0.0f;
            }
        }
        for (auto &v : state)
            if (!std::isfinite(v))
                v = 0;
        if (logger.enabled())
            logger.log(LoggerCXX::Type::COMPUTE,
                       std::string("graph iteratePropagation steps=") + std::to_string(steps) +
                           " nodes=" + std::to_string(csr.nRows) +
                           " costMs=" + std::to_string(nowEpochMs() - startMs));
        return state;
    }

    void l2NormalizeRows(std::vector<float> &data, size_t nRows, size_t nCols)
    {
        if (nRows == 0 || nCols == 0)
            return;
        for (size_t row = 0; row < nRows; row++)
        {
            double acc = 0.0;
            size_t base = row * nCols;
            for (size_t col = 0; col < nCols; col++)
            {
                double v = data[base + col];
                acc += v * v;
            }
            double norm = std::sqrt(acc);
            if (norm <= 0)
                continue;
            for (size_t col = 0; col < nCols; col++)
            {
                data[base + col] = (float)(data[base + col] / norm);
            }
        }
    }
};

// KVMStore 维护键值记忆数据的读写与检索逻辑。
// 调用方式：在会话和长期记忆路径中保存与提取状态。
// 实现思路：围绕键前缀、命名空间和索引执行操作。
// 注意事项：后端能力会影响一致性与时延表现。
// 注意事项：批量操作应明确事务边界与重试策略。
// 注意事项：该类不负责业务层权限校验。
class KVMStore
// KVMStore 维护键值记忆数据的读写与检索逻辑。
// 调用方式：在会话和长期记忆路径中保存与提取状态。
// 实现思路：围绕键前缀、命名空间和索引执行操作。
// 注意事项：后端能力会影响一致性与时延表现。
// 注意事项：批量操作应明确事务边界与重试策略。
// 注意事项：该类不负责业务层权限校验。
{
public:
    explicit KVMStore(std::shared_ptr<KeyValueStore> store, int maxCacheEntries)
        : store_(std::move(store)), maxCacheEntries_(maxCacheEntries) {}

    std::vector<std::string> getWordMemeSet(const std::string &word)
    {
        std::string key = "word:" + word;
        auto cached = cacheGet(key);
        if (cached)
            return cached->order;
        auto out = loadOrdered(key);
        return out.order;
    }

    std::vector<std::string> getMemeWords(const std::string &memeId)
    {
        std::string key = "meme:" + memeId;
        auto cached = cacheGet(key);
        if (cached)
            return cached->order;
        auto out = loadOrdered(key);
        return out.order;
    }

    void link(const std::string &word, const std::string &memeId)
    {
        std::string wordKey = "word:" + word;
        std::string memeKey = "meme:" + memeId;
        auto wordEntry = loadOrdered(wordKey);
        auto memeEntry = loadOrdered(memeKey);
        if (!wordEntry.set.count(memeId))
        {
            wordEntry.set.insert(memeId);
            wordEntry.order.push_back(memeId);
            store_->put(wordKey, json(wordEntry.order));
            cacheSet(wordKey, wordEntry);
        }
        if (!memeEntry.set.count(word))
        {
            memeEntry.set.insert(word);
            memeEntry.order.push_back(word);
            store_->put(memeKey, json(memeEntry.order));
            cacheSet(memeKey, memeEntry);
        }
    }

    void unlink(const std::string &word, const std::string &memeId)
    {
        std::string wordKey = "word:" + word;
        std::string memeKey = "meme:" + memeId;
        auto wordEntry = loadOrdered(wordKey);
        auto memeEntry = loadOrdered(memeKey);
        if (wordEntry.set.erase(memeId))
        {
            auto it = std::find(wordEntry.order.begin(), wordEntry.order.end(), memeId);
            if (it != wordEntry.order.end())
                wordEntry.order.erase(it);
            store_->put(wordKey, json(wordEntry.order));
            cacheSet(wordKey, wordEntry);
        }
        if (memeEntry.set.erase(word))
        {
            auto it = std::find(memeEntry.order.begin(), memeEntry.order.end(), word);
            if (it != memeEntry.order.end())
                memeEntry.order.erase(it);
            store_->put(memeKey, json(memeEntry.order));
            cacheSet(memeKey, memeEntry);
        }
    }

    json exportEntries()
    {
        json out;
        json words = json::object();
        json memes = json::object();
        for (auto &kv : cache_)
        {
            if (kv.first.rfind("word:", 0) == 0)
            {
                words[kv.first.substr(5)] = json(kv.second.value.order);
            }
            else if (kv.first.rfind("meme:", 0) == 0)
            {
                memes[kv.first.substr(5)] = json(kv.second.value.order);
            }
        }
        out["words"] = words;
        out["memes"] = memes;
        return out;
    }

private:
    std::shared_ptr<KeyValueStore> store_;
    struct OrderedSet
    {
        std::vector<std::string> order;
        std::unordered_set<std::string> set;
    };
    struct CacheEntry
    {
        OrderedSet value;
        std::list<std::string>::iterator it;
    };
    std::unordered_map<std::string, CacheEntry> cache_;
    std::list<std::string> lru_;
    int maxCacheEntries_{50000};

    std::optional<OrderedSet> cacheGet(const std::string &key)
    {
        if (maxCacheEntries_ <= 0)
            return std::nullopt;
        auto it = cache_.find(key);
        if (it == cache_.end())
            return std::nullopt;
        lru_.erase(it->second.it);
        lru_.push_back(key);
        it->second.it = std::prev(lru_.end());
        return it->second.value;
    }

    void cacheSet(const std::string &key, const OrderedSet &value)
    {
        if (maxCacheEntries_ <= 0)
            return;
        auto it = cache_.find(key);
        if (it != cache_.end())
        {
            it->second.value = value;
            lru_.erase(it->second.it);
            lru_.push_back(key);
            it->second.it = std::prev(lru_.end());
        }
        else
        {
            lru_.push_back(key);
            cache_[key] = CacheEntry{value, std::prev(lru_.end())};
        }
        while ((int)cache_.size() > maxCacheEntries_)
        {
            auto oldest = lru_.front();
            lru_.pop_front();
            cache_.erase(oldest);
        }
    }

    OrderedSet loadOrdered(const std::string &key)
    {
        auto cached = cacheGet(key);
        if (cached)
            return *cached;
        OrderedSet out;
        auto v = store_->get(key);
        if (v && v->is_array())
        {
            for (auto &x : *v)
            {
                if (!x.is_string())
                    continue;
                std::string s = x.get<std::string>();
                if (out.set.insert(s).second)
                    out.order.push_back(s);
            }
        }
        cacheSet(key, out);
        return out;
    }
};

// MemeGraph 表示记忆图谱的节点关系与检索接口。
// 调用方式：在学习、推理与可视化导出路径中共享。
// 实现思路：维护节点、边和索引并支持增量更新。
// 注意事项：图规模增长时需控制遍历与存储开销。
// 注意事项：并发写入应通过内部锁保证一致性。
// 注意事项：导出前建议执行数据清洗。
class MemeGraph
// MemeGraph 表示记忆图谱的节点关系与检索接口。
// 调用方式：在学习、推理与可视化导出路径中共享。
// 实现思路：维护节点、边和索引并支持增量更新。
// 注意事项：图规模增长时需控制遍历与存储开销。
// 注意事项：并发写入应通过内部锁保证一致性。
// 注意事项：导出前建议执行数据清洗。
{
public:
    // MemeGraph 构造函数绑定底层存储实例。
    // 调用方式：创建图对象时注入 KeyValueStore。
    // 实现思路：移动保存共享指针避免额外拷贝。
    // 注意事项：store 为空会影响后续持久化行为。
    explicit MemeGraph(std::shared_ptr<KeyValueStore> store) : store_(std::move(store)) {}

    void ensureNode(const std::string &memeId)
    {
        if (!nodes_.count(memeId))
        {
            nodes_[memeId] = NodeEdges{};
            nodeOrder_.push_back(memeId);
        }
        if (!meta_.count(memeId))
        {
            json meta;
            meta["degree"] = 0;
            meta["lastTouched"] = (int64_t)std::chrono::system_clock::now().time_since_epoch().count();
            meta_[memeId] = meta;
            metaOrder_.push_back(memeId);
            store_->put("node:" + memeId, meta);
        }
        if (!store_->get("row:" + memeId))
        {
            store_->put("row:" + memeId, json{{"neighbors", json::array()}});
        }
    }

    void link(const std::string &fromId, const std::string &toId, double weight = 1, int direction = 0)
    {
        if (fromId == toId)
            return;
        ensureNode(fromId);
        ensureNode(toId);
        auto &table = nodes_[fromId];
        auto it = table.map.find(toId);
        if (it == table.map.end())
        {
            table.order.push_back(toId);
            table.map[toId] = {weight, direction};
        }
        else
        {
            it->second.weight += weight;
            it->second.direction = direction;
        }
        meta_[fromId]["degree"] = (int)table.map.size();
        meta_[fromId]["lastTouched"] = (int64_t)std::chrono::system_clock::now().time_since_epoch().count();
        persistNode(fromId);
        if (direction == 0)
        {
            auto &back = nodes_[toId];
            auto it2 = back.map.find(fromId);
            if (it2 == back.map.end())
            {
                back.order.push_back(fromId);
                back.map[fromId] = {weight, direction};
            }
            else
            {
                it2->second.weight += weight;
                it2->second.direction = direction;
            }
            meta_[toId]["degree"] = (int)back.map.size();
            meta_[toId]["lastTouched"] = (int64_t)std::chrono::system_clock::now().time_since_epoch().count();
            persistNode(toId);
        }
    }

    void decayEdge(const std::string &fromId, const std::string &toId, double factor)
    {
        auto table = ensureRowLoaded(fromId);
        if (!table)
            return;
        auto it = table->map.find(toId);
        if (it == table->map.end())
            return;
        it->second.weight *= factor;
        if (it->second.weight < 1e-5)
        {
            table->map.erase(it);
            table->order.erase(std::remove(table->order.begin(), table->order.end(), toId), table->order.end());
        }
        meta_[fromId]["degree"] = (int)table->map.size();
        meta_[fromId]["lastTouched"] = (int64_t)std::chrono::system_clock::now().time_since_epoch().count();
        persistNode(fromId);
    }

    std::vector<std::string> getAllPoints()
    {
        loadAll();
        std::vector<std::string> ids;
        ids.reserve(metaOrder_.size());
        for (auto &id : metaOrder_)
        {
            if (meta_.count(id))
                ids.push_back(id);
        }
        return ids;
    }

    struct WindowInfo
    {
        CSRMatrix csr;
        std::unordered_map<std::string, uint32_t> index;
        std::vector<std::string> ids;
    };

    WindowInfo buildWindow(const std::vector<std::string> &seedIds, int radius = 1)
    {
        std::unordered_set<std::string> visited;
        std::vector<std::string> ids;
        ids.reserve(seedIds.size());
        for (auto &id : seedIds)
        {
            if (visited.insert(id).second)
                ids.push_back(id);
        }
        std::vector<std::string> border = ids;
        int depth = 0;
        while (!border.empty() && visited.size() < 4096 && depth < radius)
        {
            std::vector<std::string> next;
            for (const auto &id : border)
            {
                auto table = ensureRowLoaded(id);
                if (!table)
                    continue;
                for (auto &to : table->order)
                {
                    if (visited.insert(to).second)
                    {
                        next.push_back(to);
                        ids.push_back(to);
                    }
                }
            }
            border = next;
            depth++;
        }
        std::unordered_map<std::string, uint32_t> index;
        for (uint32_t i = 0; i < ids.size(); i++)
            index[ids[i]] = i;
        CSRMatrix csr;
        csr.nRows = csr.nCols = (uint32_t)ids.size();
        csr.rowPtr.resize(ids.size() + 1);
        for (uint32_t i = 0; i < ids.size(); i++)
        {
            auto table = ensureRowLoaded(ids[i]);
            csr.rowPtr[i] = (uint32_t)csr.colIdx.size();
            if (table)
            {
                for (auto &to : table->order)
                {
                    auto itEdge = table->map.find(to);
                    if (itEdge == table->map.end())
                        continue;
                    auto it = index.find(to);
                    if (it == index.end())
                        continue;
                    csr.colIdx.push_back(it->second);
                    csr.values.push_back((float)itEdge->second.weight);
                }
            }
        }
        csr.rowPtr[ids.size()] = (uint32_t)csr.colIdx.size();
        return {csr, index, ids};
    }

    std::vector<std::vector<float>> messagePassingEmbeddings(const std::vector<std::string> &ids,
                                                             const std::vector<std::vector<float>> &baseEmb,
                                                             int rounds = 1,
                                                             float selfWeight = 0.6f,
                                                             float neighborWeight = 0.4f)
    {
        if (ids.empty() || baseEmb.empty())
            return baseEmb;
        std::unordered_map<std::string, size_t> index;
        for (size_t i = 0; i < ids.size(); i++)
            index[ids[i]] = i;
        std::vector<std::vector<float>> cur = baseEmb;
        for (int r = 0; r < rounds; r++)
        {
            std::vector<std::vector<float>> next = cur;
            for (size_t i = 0; i < ids.size(); i++)
            {
                auto table = ensureRowLoaded(ids[i]);
                if (!table)
                    continue;
                std::vector<float> agg(cur[i].size(), 0.0f);
                double wsum = 0.0;
                for (auto &to : table->order)
                {
                    auto it = table->map.find(to);
                    if (it == table->map.end())
                        continue;
                    auto itIdx = index.find(to);
                    if (itIdx == index.end())
                        continue;
                    double w = it->second.weight;
                    if (it->second.direction == 1)
                        w *= 0.85; // inbound
                    else if (it->second.direction == 2)
                        w *= 1.15; // outbound
                    for (size_t d = 0; d < agg.size() && d < cur[itIdx->second].size(); d++)
                    {
                        agg[d] += (float)(w * cur[itIdx->second][d]);
                    }
                    wsum += std::abs(w);
                }
                if (wsum > 0.0)
                {
                    for (auto &v : agg)
                        v = (float)(v / wsum);
                }
                for (size_t d = 0; d < next[i].size() && d < agg.size(); d++)
                {
                    next[i][d] = selfWeight * cur[i][d] + neighborWeight * agg[d];
                }
                float norm = 0.0f;
                for (float v : next[i])
                    norm += v * v;
                norm = std::sqrt(norm);
                if (norm > 0.0f)
                    for (auto &v : next[i])
                        v /= norm;
            }
            cur = std::move(next);
        }
        return cur;
    }

    json exportSnapshot()
    {
        loadAll();
        json nodes = json::array();
        for (auto &id : nodeOrder_)
        {
            auto it = nodes_.find(id);
            if (it == nodes_.end())
                continue;
            json neighbors = json::array();
            for (auto &to : it->second.order)
            {
                auto itEdge = it->second.map.find(to);
                if (itEdge == it->second.map.end())
                    continue;
                neighbors.push_back({{"to", to}, {"weight", itEdge->second.weight}, {"direction", itEdge->second.direction}});
            }
            nodes.push_back({{"id", id}, {"neighbors", neighbors}});
        }
        json meta = json::object();
        for (auto &id : metaOrder_)
        {
            auto it = meta_.find(id);
            if (it != meta_.end())
                meta[id] = it->second;
        }
        return {{"nodes", nodes}, {"meta", meta}};
    }

    std::vector<std::pair<std::string, std::pair<double, int>>> neighborsOf(const std::string &memeId)
    {
        auto table = ensureRowLoaded(memeId);
        std::vector<std::pair<std::string, std::pair<double, int>>> out;
        if (!table)
            return out;
        for (auto &to : table->order)
        {
            auto it = table->map.find(to);
            if (it == table->map.end())
                continue;
            out.push_back({to, {it->second.weight, it->second.direction}});
        }
        return out;
    }

    int degreeOf(const std::string &memeId)
    {
        auto table = ensureRowLoaded(memeId);
        return table ? (int)table->map.size() : 0;
    }

    int removeOutgoingEdges(const std::string &memeId)
    {
        auto tablePtr = ensureRowLoaded(memeId);
        if (!tablePtr)
            return 0;
        auto &table = nodes_[memeId];
        int cut = 0;
        for (auto it = table.order.begin(); it != table.order.end();)
        {
            auto itEdge = table.map.find(*it);
            if (itEdge != table.map.end() && itEdge->second.direction == 2)
            {
                std::string targetId = *it;
                table.map.erase(itEdge);
                it = table.order.erase(it);
                cut++;
                auto targetTable = ensureRowLoaded(targetId);
                if (targetTable)
                {
                    auto &t = nodes_[targetId];
                    for (auto it2 = t.order.begin(); it2 != t.order.end();)
                    {
                        auto itEdge2 = t.map.find(*it2);
                        if (itEdge2 != t.map.end() && *it2 == memeId && itEdge2->second.direction == 1)
                        {
                            t.map.erase(itEdge2);
                            it2 = t.order.erase(it2);
                        }
                        else
                        {
                            ++it2;
                        }
                    }
                    meta_[targetId]["degree"] = (int)t.map.size();
                    meta_[targetId]["lastTouched"] = (int64_t)std::chrono::system_clock::now().time_since_epoch().count();
                    persistNode(targetId);
                }
            }
            else
            {
                ++it;
            }
        }
        meta_[memeId]["degree"] = (int)table.map.size();
        meta_[memeId]["lastTouched"] = (int64_t)std::chrono::system_clock::now().time_since_epoch().count();
        persistNode(memeId);
        return cut;
    }
    bool removeNode(const std::string &memeId)
    {
        loadAll();
        if (!nodes_.count(memeId))
            return false;
        nodes_.erase(memeId);
        store_->del("row:" + memeId);
        meta_.erase(memeId);
        store_->del("node:" + memeId);
        nodeOrder_.erase(std::remove(nodeOrder_.begin(), nodeOrder_.end(), memeId), nodeOrder_.end());
        metaOrder_.erase(std::remove(metaOrder_.begin(), metaOrder_.end(), memeId), metaOrder_.end());
        for (auto &kv : nodes_)
        {
            auto &table = kv.second;
            if (table.map.erase(memeId))
            {
                table.order.erase(std::remove(table.order.begin(), table.order.end(), memeId), table.order.end());
                meta_[kv.first]["degree"] = (int)table.map.size();
                meta_[kv.first]["lastTouched"] = (int64_t)std::chrono::system_clock::now().time_since_epoch().count();
                persistNode(kv.first);
            }
        }
        return true;
    }

    json getMeta(const std::string &memeId)
    {
        loadAll();
        auto it = meta_.find(memeId);
        return it == meta_.end() ? json::object() : it->second;
    }

    void setMeta(const std::string &memeId, const json &meta)
    {
        if (!meta_.count(memeId))
            metaOrder_.push_back(memeId);
        meta_[memeId] = meta;
        store_->put("node:" + memeId, meta);
    }

    std::vector<std::pair<std::string, json>> metaEntries()
    {
        loadAll();
        std::vector<std::pair<std::string, json>> out;
        out.reserve(meta_.size());
        for (auto &id : metaOrder_)
        {
            auto it = meta_.find(id);
            if (it != meta_.end())
                out.push_back({id, it->second});
        }
        return out;
    }

    void importSnapshot(const json &snapshot)
    {
        nodes_.clear();
        meta_.clear();
        nodeOrder_.clear();
        metaOrder_.clear();
        for (auto &item : snapshot.value("nodes", json::array()))
        {
            std::string id = item.value("id", "");
            if (id.empty())
                continue;
            NodeEdges node;
            for (auto &n : item.value("neighbors", json::array()))
            {
                std::string to = n.value("to", "");
                if (to.empty())
                    continue;
                if (node.map.find(to) == node.map.end())
                    node.order.push_back(to);
                node.map[to] = {n.value("weight", 0.0), n.value("direction", 0)};
            }
            nodes_[id] = std::move(node);
            nodeOrder_.push_back(id);
            persistNode(id);
        }
        json meta = snapshot.value("meta", json::object());
        for (auto it = meta.begin(); it != meta.end(); ++it)
        {
            meta_[it.key()] = it.value();
            metaOrder_.push_back(it.key());
            store_->put("node:" + it.key(), it.value());
        }
    }

private:
    struct Edge
    {
        double weight{0};
        int direction{0};
    };
    struct NodeEdges
    {
        std::vector<std::string> order;
        std::unordered_map<std::string, Edge> map;
    };
    std::shared_ptr<KeyValueStore> store_;
    std::unordered_map<std::string, NodeEdges> nodes_;
    std::vector<std::string> nodeOrder_;
    std::unordered_map<std::string, json> meta_;
    std::vector<std::string> metaOrder_;
    bool fullyLoaded_{false};

    void loadAll()
    {
        if (fullyLoaded_)
            return;
        auto metaEntries = store_->entries("node:");
        for (auto &kv : metaEntries)
        {
            std::string id = kv.first.substr(5);
            meta_[id] = kv.second;
            if (std::find(metaOrder_.begin(), metaOrder_.end(), id) == metaOrder_.end())
                metaOrder_.push_back(id);
        }
        auto rowEntries = store_->entries("row:");
        for (auto &kv : rowEntries)
        {
            std::string id = kv.first.substr(4);
            NodeEdges node;
            for (auto &n : kv.second.value("neighbors", json::array()))
            {
                std::string to = n.value("to", "");
                if (to.empty())
                    continue;
                if (node.map.find(to) == node.map.end())
                    node.order.push_back(to);
                node.map[to] = {n.value("weight", 0.0), n.value("direction", 0)};
            }
            if (std::find(nodeOrder_.begin(), nodeOrder_.end(), id) == nodeOrder_.end())
                nodeOrder_.push_back(id);
            nodes_[id] = std::move(node);
        }
        fullyLoaded_ = true;
    }

    NodeEdges *ensureRowLoaded(const std::string &id)
    {
        auto it = nodes_.find(id);
        if (it != nodes_.end())
            return &it->second;
        auto row = store_->get("row:" + id);
        if (row && row->is_object())
        {
            NodeEdges node;
            for (auto &n : row->value("neighbors", json::array()))
            {
                std::string to = n.value("to", "");
                if (to.empty())
                    continue;
                if (node.map.find(to) == node.map.end())
                    node.order.push_back(to);
                node.map[to] = {n.value("weight", 0.0), n.value("direction", 0)};
            }
            nodes_[id] = std::move(node);
            if (std::find(nodeOrder_.begin(), nodeOrder_.end(), id) == nodeOrder_.end())
                nodeOrder_.push_back(id);
            return &nodes_[id];
        }
        return nullptr;
    }

    void persistNode(const std::string &id)
    {
        auto &table = nodes_[id];
        json neighbors = json::array();
        for (auto &to : table.order)
        {
            auto it = table.map.find(to);
            if (it == table.map.end())
                continue;
            neighbors.push_back({{"to", to}, {"weight", it->second.weight}, {"direction", it->second.direction}});
        }
        store_->put("row:" + id, json{{"neighbors", neighbors}});
    }
};

class GraphExportBuilder
// GraphExportBuilder 负责将图数据转换为导出格式。
// 调用方式：提供文件路径或流对象后构建导出内容。
// 实现思路：按节点与边组织结构并附带属性信息。
// 注意事项：大规模导出建议分段写入降低峰值内存。
// 注意事项：缺失字段需有稳健默认值。
// 注意事项：该类不修改源图数据。
{
public:
    static json fromWindow(const MemeGraph::WindowInfo &windowInfo, const std::vector<float> *activation = nullptr)
    {
        const auto &ids = windowInfo.ids;
        int n = (int)ids.size();
        json nodes = json::array();
        json edges = json::array();
        for (int i = 0; i < n; i++)
        {
            double ax = (activation && i < (int)activation->size() && std::isfinite((*activation)[i])) ? (*activation)[i] : 0.0;
            double ay = ((i * 9973) % 101) / 100.0;
            nodes.push_back({{"id", ids[i]},
                             {"x", ax},
                             {"y", ay},
                             {"z", 0.0},
                             {"value", ax},
                             {"attrs", json::object()}});
        }
        const auto &csr = windowInfo.csr;
        for (uint32_t r = 0; r < csr.nRows; r++)
        {
            uint32_t start = csr.rowPtr[r];
            uint32_t end = csr.rowPtr[r + 1];
            for (uint32_t k = start; k < end; k++)
            {
                uint32_t c = csr.colIdx[k];
                double w = (k < csr.values.size()) ? csr.values[k] : 0.0;
                if (r < ids.size() && c < ids.size())
                {
                    edges.push_back({{"from", ids[r]},
                                     {"to", ids[c]},
                                     {"weight", std::isfinite(w) ? w : 0.0},
                                     {"dir", 1}});
                }
            }
        }
        return json{{"Nodes", nodes}, {"Edges", edges}};
    }
};

class GraphTensorBridge
// GraphTensorBridge 负责图结构与张量表示之间的映射。
// 调用方式：训练或推理前将图特征编码为张量输入。
// 实现思路：抽取节点/边属性并构建统一数值表示。
// 注意事项：映射规则变化会影响模型兼容性。
// 注意事项：输入图缺失字段时需执行容错处理。
// 注意事项：桥接过程应尽量减少重复拷贝。
{
public:
    explicit GraphTensorBridge(KVMStore *kvm, TensorEngine *tensor)
        : kvm_(kvm), tensor_(tensor) {}

    std::vector<std::vector<float>> computeNodeEmbeddings(const std::vector<std::string> &ids, int D = 512)
    {
        rebuildRowIndex(ids);
        std::vector<std::vector<float>> embeddings(ids.size(), std::vector<float>(D, 0.0f));
        for (size_t i = 0; i < ids.size(); i++)
        {
            auto words = kvm_->getMemeWords(ids[i]);
            if (!words.empty())
            {
                std::string joined;
                for (auto &word : words)
                {
                    if (word.empty())
                        continue;
                    if (!joined.empty())
                        joined.push_back(' ');
                    joined += word;
                }
                if (!joined.empty())
                {
                    auto base = textToMiniEmbedding(joined, D);
                    if (base.size() == (size_t)D)
                    {
                        embeddings[i] = std::move(base);
                        continue;
                    }
                }
            }
            uint32_t idx = fnv1a32(ids[i]) % (uint32_t)D;
            embeddings[i][idx] += 1.0f;
        }
        // l2NormalizeRows for std::vector<std::vector<float>>
        for (auto &vec : embeddings)
        {
            float norm = 0.0f;
            for (float v : vec)
                norm += v * v;
            norm = std::sqrt(norm);
            if (norm > 0.0f)
            {
                for (float &v : vec)
                    v /= norm;
            }
        }
        return embeddings;
    }

    static uint32_t fnv1a32(const std::string &str)
    {
        uint32_t hash = 0x811c9dc5u;
        for (unsigned char c : str)
        {
            hash ^= c;
            hash *= 0x01000193u;
        }
        return hash;
    }

    void rebuildRowIndex(const std::vector<std::string> &ids)
    {
        memeIndex_.clear();
        for (size_t i = 0; i < ids.size(); i++)
            memeIndex_[ids[i]] = (uint32_t)i;
    }

    json buildEmbeddings(const std::vector<std::string> &ids, int D = 512)
    {
        rebuildRowIndex(ids);
        std::vector<float> mat(ids.size() * (size_t)D, 0.0f);
        for (size_t i = 0; i < ids.size(); i++)
        {
            auto words = kvm_->getMemeWords(ids[i]);
            if (words.empty())
                continue;
            for (auto &word : words)
            {
                uint32_t idx = fnv1a32(word) % (uint32_t)D;
                mat[i * (size_t)D + idx] += 1.0f;
            }
        }
        tensor_->l2NormalizeRows(mat, ids.size(), (size_t)D);
        embeddings_ = json{{"data", mat}, {"nRows", (int)ids.size()}, {"nCols", D}};
        return embeddings_;
    }

    json buildMultiOrderCSR(const MemeGraph::WindowInfo &graphWindow)
    {
        const auto &csr = graphWindow.csr;
        const auto &ids = graphWindow.ids;
        std::vector<uint32_t> biRow(csr.rowPtr.size(), 0);
        std::vector<uint32_t> biCol;
        std::vector<float> biVal;
        std::vector<uint32_t> inRow(csr.rowPtr.size(), 0);
        std::vector<uint32_t> inCol;
        std::vector<float> inVal;

        std::vector<std::unordered_map<uint32_t, float>> adjacency(ids.size());
        for (uint32_t row = 0; row < ids.size(); row++)
        {
            uint32_t start = csr.rowPtr[row];
            uint32_t end = csr.rowPtr[row + 1];
            for (uint32_t idx = start; idx < end; idx++)
            {
                uint32_t col = csr.colIdx[idx];
                float weight = csr.values[idx];
                adjacency[row][col] = weight;
            }
        }

        uint32_t biCounter = 0;
        for (uint32_t row = 0; row < ids.size(); row++)
        {
            biRow[row] = biCounter;
            uint32_t start = csr.rowPtr[row];
            uint32_t end = csr.rowPtr[row + 1];
            for (uint32_t idx = start; idx < end; idx++)
            {
                uint32_t col = csr.colIdx[idx];
                float weight = csr.values[idx];
                auto it = adjacency[col].find(row);
                if (it != adjacency[col].end() && it->second != 0.0f)
                {
                    biCol.push_back(col);
                    biVal.push_back((weight + it->second) * 0.5f);
                    biCounter++;
                }
                inCol.push_back(row);
                inVal.push_back(weight);
            }
        }
        biRow[ids.size()] = biCounter;
        inRow[ids.size()] = (uint32_t)inCol.size();

        json all = json{{"rowPtr", csr.rowPtr}, {"colIdx", csr.colIdx}, {"values", csr.values}, {"nRows", csr.nRows}, {"nCols", csr.nCols}};
        json bi = json{{"rowPtr", biRow}, {"colIdx", biCol}, {"values", biVal}, {"nRows", (int)ids.size()}, {"nCols", (int)ids.size()}};
        json inbound = json{{"rowPtr", inRow}, {"colIdx", inCol}, {"values", inVal}, {"nRows", (int)ids.size()}, {"nCols", (int)ids.size()}};
        multi_ = json{{"all", all}, {"bi", bi}, {"out", all}, {"in", inbound}, {"ids", ids}};
        return multi_;
    }

private:
    KVMStore *kvm_{nullptr};
    TensorEngine *tensor_{nullptr};
    std::unordered_map<std::string, uint32_t> memeIndex_;
    json embeddings_ = json::object();
    json multi_ = json::object();
};

class DimReducer
{
public:
    // PCAResult 保存主成分分析输出的核心结果。
    // 调用方式：由 PCA 计算函数返回并供后续投影使用。
    // 实现思路：聚合均值向量、主成分矩阵和特征值。
    // 注意事项：components 与 eigenValues 维度需一致。
    // 注意事项：结果依赖输入归一化与采样质量。
    // 注意事项：该结构适合作为算法中间产物传递。
    struct PCAResult
    {
        std::vector<float> coords;
        std::vector<float> v1;
        std::vector<float> v2;
    };

    PCAResult pca2D(const json &emb)
    {
        auto data = emb.value("data", std::vector<float>{});
        int nRows = emb.value("nRows", 0);
        int nCols = emb.value("nCols", 0);
        std::vector<float> mean(nCols, 0.0f);
        for (int row = 0; row < nRows; row++)
        {
            for (int col = 0; col < nCols; col++)
            {
                mean[col] += data[row * nCols + col];
            }
        }
        for (int col = 0; col < nCols; col++)
            mean[col] /= (nRows > 0 ? nRows : 1);

        std::vector<float> centered(data.size(), 0.0f);
        for (int row = 0; row < nRows; row++)
        {
            for (int col = 0; col < nCols; col++)
            {
                int idx = row * nCols + col;
                centered[idx] = data[idx] - mean[col];
            }
        }

        std::vector<std::vector<float>> cov(nCols, std::vector<float>(nCols, 0.0f));
        for (int row = 0; row < nRows; row++)
        {
            for (int i = 0; i < nCols; i++)
            {
                for (int j = i; j < nCols; j++)
                {
                    float v = centered[row * nCols + i] * centered[row * nCols + j];
                    cov[i][j] += v;
                    if (i != j)
                        cov[j][i] += v;
                }
            }
        }

        auto powerIter = [&](const std::vector<float> &vec)
        {
            std::vector<float> next(nCols, 0.0f);
            for (int i = 0; i < nCols; i++)
            {
                double acc = 0.0;
                for (int j = 0; j < nCols; j++)
                    acc += cov[i][j] * vec[j];
                next[i] = (float)acc;
            }
            double norm = 0.0;
            for (auto &v : next)
                norm += v * v;
            norm = std::sqrt(norm) ? std::sqrt(norm) : 1.0;
            for (auto &v : next)
                v = (float)(v / norm);
            return next;
        };

        std::mt19937 rng((unsigned)std::chrono::high_resolution_clock::now().time_since_epoch().count());
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
        std::vector<float> v1(nCols, 0.0f);
        for (auto &v : v1)
            v = dist(rng);
        for (int i = 0; i < 8; i++)
            v1 = powerIter(v1);
        std::vector<float> v2(nCols, 0.0f);
        for (auto &v : v2)
            v = dist(rng);
        for (int i = 0; i < 8; i++)
        {
            double proj = 0.0;
            for (int j = 0; j < nCols; j++)
                proj += v2[j] * v1[j];
            for (int j = 0; j < nCols; j++)
                v2[j] = (float)(v2[j] - proj * v1[j]);
            v2 = powerIter(v2);
        }

        std::vector<float> coords(nRows * 2, 0.0f);
        for (int row = 0; row < nRows; row++)
        {
            double x = 0.0;
            double y = 0.0;
            for (int col = 0; col < nCols; col++)
            {
                float v = centered[row * nCols + col];
                x += v * v1[col];
                y += v * v2[col];
            }
            coords[row * 2] = (float)x;
            coords[row * 2 + 1] = (float)y;
        }
        return {coords, v1, v2};
    }

    json project2D(const json &emb, const std::string &method = "auto")
    {
        if (method == "umap")
        {
            auto coords2d = umap2D(emb, 15, 200);
            json coords = json::array();
            for (auto &c : coords2d)
                coords.push_back(json::array({c[0], c[1]}));
            return json{{"coords", coords}, {"method", "umap"}};
        }
        auto pca = pca2D(emb);
        int nRows = emb.value("nRows", 0);
        json coords = json::array();
        for (int row = 0; row < nRows; row++)
        {
            coords.push_back(json::array({pca.coords[row * 2], pca.coords[row * 2 + 1]}));
        }
        return json{{"coords", coords}, {"method", "pca"}};
    }

private:
    static std::vector<std::array<float, 2>> umap2D(const json &emb, int nNeighbors, int epochs)
    {
        int nRows = emb.value("nRows", 0);
        int nCols = emb.value("nCols", 0);
        if (nRows <= 0 || nCols <= 0)
            return {};
        std::vector<float> data;
        if (emb.contains("data") && emb["data"].is_array())
        {
            data.reserve((size_t)nRows * (size_t)nCols);
            for (auto &v : emb["data"])
                data.push_back(v.get<float>());
        }
        if ((int)data.size() != nRows * nCols)
            return {};

        int k = std::max(2, std::min(nNeighbors, nRows - 1));
        double target = std::log2((double)k);

        std::vector<std::vector<std::pair<int, float>>> knn(nRows);
        for (int i = 0; i < nRows; i++)
        {
            std::vector<std::pair<float, int>> dists;
            dists.reserve(nRows - 1);
            for (int j = 0; j < nRows; j++)
            {
                if (i == j)
                    continue;
                double acc = 0.0;
                for (int c = 0; c < nCols; c++)
                {
                    double diff = data[i * nCols + c] - data[j * nCols + c];
                    acc += diff * diff;
                }
                dists.push_back({(float)std::sqrt(acc), j});
            }
            std::partial_sort(dists.begin(), dists.begin() + k, dists.end(), [](auto &a, auto &b)
                              { return a.first < b.first; });
            for (int t = 0; t < k; t++)
                knn[i].push_back({dists[t].second, dists[t].first});
        }

        std::vector<float> rhos(nRows, 0.0f);
        std::vector<float> sigmas(nRows, 1.0f);
        for (int i = 0; i < nRows; i++)
        {
            float rho = 0.0f;
            for (auto &p : knn[i])
            {
                if (p.second > 0)
                {
                    rho = p.second;
                    break;
                }
            }
            rhos[i] = rho;
            double lo = 1e-3, hi = 64.0;
            for (int it = 0; it < 32; it++)
            {
                double mid = (lo + hi) * 0.5;
                double sum = 0.0;
                for (auto &p : knn[i])
                {
                    double d = std::max(0.0, (double)p.second - rho);
                    sum += std::exp(-d / mid);
                }
                if (sum > target)
                    hi = mid;
                else
                    lo = mid;
            }
            sigmas[i] = (float)((lo + hi) * 0.5);
            if (sigmas[i] <= 1e-6f)
                sigmas[i] = 1.0f;
        }

        std::vector<std::unordered_map<int, float>> weights(nRows);
        for (int i = 0; i < nRows; i++)
        {
            for (auto &p : knn[i])
            {
                double d = std::max(0.0, (double)p.second - rhos[i]);
                double val = std::exp(-d / sigmas[i]);
                weights[i][p.first] = (float)val;
            }
        }

        struct Edge
        {
            int i;
            int j;
            float w;
        };
        std::vector<Edge> edges;
        edges.reserve(nRows * k);
        for (int i = 0; i < nRows; i++)
        {
            for (auto &kv : weights[i])
            {
                int j = kv.first;
                if (i >= j)
                    continue;
                float w1 = kv.second;
                float w2 = weights[j].count(i) ? weights[j][i] : 0.0f;
                float ws = w1 + w2 - w1 * w2;
                if (ws > 0.0f)
                    edges.push_back({i, j, ws});
            }
        }

        auto find_ab = [&](double spread, double minDist)
        {
            double bestA = 1.0, bestB = 1.0, bestErr = 1e9;
            double x1 = minDist + spread;
            double x2 = minDist + 2.0 * spread;
            double t1 = std::exp(-1.0);
            double t2 = std::exp(-2.0);
            for (double b = 0.5; b <= 2.5; b += 0.01)
            {
                double a = (1.0 / t1 - 1.0) / std::pow(x1, 2.0 * b);
                if (!std::isfinite(a) || a <= 0)
                    continue;
                double f2 = 1.0 / (1.0 + a * std::pow(x2, 2.0 * b));
                double err = std::abs(f2 - t2);
                if (err < bestErr)
                {
                    bestErr = err;
                    bestA = a;
                    bestB = b;
                }
            }
            return std::pair<double, double>(bestA, bestB);
        };

        double minDist = 0.1;
        double spread = 1.0;
        auto ab = find_ab(spread, minDist);
        double a = ab.first;
        double b = ab.second;
        double gamma = 1.0;

        DimReducer reducer;
        auto pca = reducer.pca2D(emb);
        std::vector<std::array<float, 2>> coords(nRows, {0.0f, 0.0f});
        if ((int)pca.coords.size() >= nRows * 2)
        {
            for (int i = 0; i < nRows; i++)
            {
                coords[i][0] = pca.coords[i * 2];
                coords[i][1] = pca.coords[i * 2 + 1];
            }
        }
        else
        {
            std::mt19937 rng(0xBADC0DEu);
            std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
            for (int i = 0; i < nRows; i++)
            {
                coords[i][0] = dist(rng);
                coords[i][1] = dist(rng);
            }
        }

        std::mt19937 rng(0x12345678u);
        std::uniform_int_distribution<int> pick(0, nRows - 1);
        int negSamples = 5;
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            double lr = 1.0 * (1.0 - (double)epoch / std::max(1, epochs));
            for (const auto &e : edges)
            {
                auto &pa = coords[e.i];
                auto &pb = coords[e.j];
                double dx = pa[0] - pb[0];
                double dy = pa[1] - pb[1];
                double dist2 = dx * dx + dy * dy + 1e-6;
                double gradCoeff = -2.0 * a * b * std::pow(dist2, b - 1.0) / (1.0 + a * std::pow(dist2, b));
                double step = lr * e.w * gradCoeff;
                pa[0] += (float)(step * dx);
                pa[1] += (float)(step * dy);
                pb[0] -= (float)(step * dx);
                pb[1] -= (float)(step * dy);

                for (int t = 0; t < negSamples; t++)
                {
                    int r = pick(rng);
                    if (r == e.i)
                        continue;
                    auto &pn = coords[r];
                    double ndx = pa[0] - pn[0];
                    double ndy = pa[1] - pn[1];
                    double nd2 = ndx * ndx + ndy * ndy + 1e-6;
                    double repCoeff = 2.0 * gamma * b / ((0.001 + nd2) * (1.0 + a * std::pow(nd2, b)));
                    double rep = lr * repCoeff;
                    pa[0] += (float)(rep * ndx);
                    pa[1] += (float)(rep * ndy);
                    pn[0] -= (float)(rep * ndx);
                    pn[1] -= (float)(rep * ndy);
                }
            }
        }
        return coords;
    }
};

class PatternMatrix
{
public:
    // PatternMatrix 构造函数注入图桥接所需依赖。
    // 调用方式：运行时初始化阶段创建一次即可。
    // 实现思路：内部组合 GraphTensorBridge 对象。
    // 注意事项：依赖对象生命周期需长于本实例。
    PatternMatrix(KVMStore *kvm, TensorEngine *tensor)
        : bridge_(kvm, tensor) {}

    json rebuild(const MemeGraph::WindowInfo &windowInfo)
    {
        auto emb = bridge_.buildEmbeddings(windowInfo.ids, 512);
        auto multi = bridge_.buildMultiOrderCSR(windowInfo);
        auto projection = reducer_.project2D(emb, "auto");
        state_ = json{{"ids", windowInfo.ids}, {"embeddings", emb}, {"multi", multi}, {"projection", projection}};
        return state_;
    }

    // state 返回最近一次重建后的矩阵状态。
    // 调用方式：用于调试可视化或接口下发。
    // 实现思路：直接返回缓存的 JSON 快照。
    // 注意事项：返回值是副本，修改不会回写内部状态。
    json state() const { return state_; }

private:
    GraphTensorBridge bridge_;
    DimReducer reducer_;
    json state_ = json::object();
};

// MemeSurfaceLexicon 管理 meme 到表层短语的映射词典。
// 调用方式：学习阶段调用 learn，推理阶段调用 recall。
// 实现思路：按衰减策略维护短语权重并限制条目规模。
// 注意事项：同一 meme 的高频写入需关注存储压力。
// 注意事项：权重阈值与衰减参数会影响召回稳定性。
// 注意事项：该类不直接决定最终回复文本。
class MemeSurfaceLexicon
{
public:
    MemeSurfaceLexicon(std::shared_ptr<KeyValueStore> store, int maxEntriesPerMeme = 64, double decay = 0.985)
        : store_(std::move(store)), maxEntries_(std::max(8, maxEntriesPerMeme)), decay_(decay) {}

    void learn(const std::string &memeId, const std::string &replyText, double weight = 1.0)
    {
        if (memeId.empty())
            return;
        auto rec = load(memeId);
        json phrases = rec.value("phrases", json::object());
        for (auto it = phrases.begin(); it != phrases.end();)
        {
            double v = 0.0;
            if (it.value().is_number())
            {
                v = it.value().get<double>();
            }
            else if (it.value().is_string())
            {
                try
                {
                    v = std::stod(it.value().get<std::string>());
                }
                catch (...)
                {
                    v = 0.0;
                }
            }
            double nv = v * decay_;
            if (nv <= 1e-6)
            {
                it = phrases.erase(it);
            }
            else
            {
                it.value() = nv;
                ++it;
            }
        }
        for (auto &p : extractSurfacePhrases(replyText, 24))
        {
            double prev = phrases.value(p.phrase, 0.0);
            phrases[p.phrase] = prev + p.weight * weight;
        }
        std::vector<std::pair<std::string, double>> ordered;
        for (auto it = phrases.begin(); it != phrases.end(); ++it)
        {
            if (it.value().is_number())
                ordered.push_back({it.key(), it.value().get<double>()});
        }
        std::sort(ordered.begin(), ordered.end(), [](auto &a, auto &b)
                  { return a.second > b.second; });
        if ((int)ordered.size() > maxEntries_)
            ordered.resize(maxEntries_);
        json next;
        next["updatedAt"] = (int64_t)std::chrono::system_clock::now().time_since_epoch().count();
        json obj = json::object();
        for (auto &kv : ordered)
            obj[kv.first] = kv.second;
        next["phrases"] = obj;
        store_->put("m:" + memeId, next);
    }

    std::vector<SurfacePhrase> getTop(const std::string &memeId, int limit = 6)
    {
        auto rec = load(memeId);
        std::vector<SurfacePhrase> out;
        json phrases = rec.value("phrases", json::object());
        for (auto it = phrases.begin(); it != phrases.end(); ++it)
        {
            double v = 0.0;
            if (it.value().is_number())
            {
                v = it.value().get<double>();
            }
            else if (it.value().is_string())
            {
                try
                {
                    v = std::stod(it.value().get<std::string>());
                }
                catch (...)
                {
                    v = 0.0;
                }
            }
            out.push_back({it.key(), v});
        }
        std::sort(out.begin(), out.end(), [](auto &a, auto &b)
                  { return a.weight > b.weight; });
        if ((int)out.size() > limit)
            out.resize(limit);
        return out;
    }

    json exportSnapshot(int limitMemes = 512)
    {
        json out = json::array();
        auto items = store_->entries("m:");
        for (auto &kv : items)
        {
            out.push_back({kv.first, kv.second});
            if ((int)out.size() >= limitMemes)
                break;
        }
        return out;
    }

    void importSnapshot(const json &entries)
    {
        if (!entries.is_array())
            return;
        for (auto &item : entries)
        {
            if (!item.is_array() || item.size() != 2)
                continue;
            std::string key = item[0].get<std::string>();
            if (key.rfind("m:", 0) != 0)
                continue;
            store_->put(key, item[1]);
        }
    }

private:
    std::shared_ptr<KeyValueStore> store_;
    int maxEntries_;
    double decay_;

    json load(const std::string &memeId)
    {
        auto v = store_->get("m:" + memeId);
        if (v && v->is_object())
            return *v;
        return json{{"phrases", json::object()}, {"updatedAt", 0}};
    }
};

// DialogMemory 负责存储和检索对话记忆片段。
// 调用方式：对话完成后写入，召回时按签名或 meme 检索。
// 实现思路：维护主记录与倒排索引并控制容量。
// 注意事项：reply/问题文本长度会被裁剪以控制体积。
// 注意事项：索引键冲突需通过稳定哈希策略规避。
// 注意事项：该组件应与业务隐私策略配套使用。
class DialogMemory
{
public:
    DialogMemory(std::shared_ptr<KeyValueStore> store, int maxItems = 2048, int maxPerIndex = 64)
        : store_(std::move(store)), maxItems_(std::max(128, maxItems)), maxPerIndex_(std::max(8, maxPerIndex)) {}

    json remember(const std::string &sessionId,
                 const std::string &signature,
                 const std::vector<std::string> &memes,
                 const std::string &question,
                 const std::string &reply,
                 double scoreHint = 0.0)
    {
        std::string sid = scopeKey(sessionId);
        if (sid.empty() || signature.empty() || reply.empty())
            return json();
        std::string id = sid + ":" + sha1Hex(signature);
        auto key = "d:" + id;
        auto prev = store_->get(key);
        int count = prev ? prev->value("count", 0) : 0;
        double prevHint = prev ? prev->value("scoreHint", 0.0) : 0.0;
        double hint = std::isfinite(scoreHint) ? scoreHint : prevHint;
        json rec = {
            {"id", id}, {"sessionId", sid}, {"signature", signature}, {"memes", memes}, {"question", question.substr(0, 800)}, {"reply", reply.substr(0, 1200)}, {"updatedAt", (int64_t)std::chrono::system_clock::now().time_since_epoch().count()}, {"count", count + 1}, {"scoreHint", hint}};
        store_->put(key, rec);
        for (const auto &memeId : std::unordered_set<std::string>(memes.begin(), memes.end()))
        {
            std::string ik = "i:" + sid + ":" + memeId;
            auto ids = store_->get(ik);
            std::vector<std::string> list;
            if (ids && ids->is_array())
            {
                for (auto &x : *ids)
                {
                    if (x.is_string())
                        list.push_back(x.get<std::string>());
                }
            }
            list.erase(std::remove(list.begin(), list.end(), id), list.end());
            list.insert(list.begin(), id);
            if ((int)list.size() > maxPerIndex_)
                list.resize(maxPerIndex_);
            store_->put(ik, list);
        }
        return rec;
    }

    json retrieve(const std::string &sessionId, const std::vector<std::string> &memes, const std::string &signature, double minSim = 0.45)
    {
        std::string sid = scopeKey(sessionId);
        if (sid.empty())
            return json();
        std::unordered_set<std::string> candidateIds;
        for (size_t i = 0; i < std::min<size_t>(8, memes.size()); i++)
        {
            auto ids = store_->get("i:" + sid + ":" + memes[i]);
            if (ids && ids->is_array())
            {
                for (auto &x : *ids)
                {
                    if (x.is_string() && !x.get<std::string>().empty())
                        candidateIds.insert(x.get<std::string>());
                }
            }
        }
        if (candidateIds.empty() && !signature.empty())
            candidateIds.insert(sid + ":" + sha1Hex(signature));

        auto toSet = [](const std::string &sig)
        {
            std::unordered_set<std::string> out;
            std::stringstream ss(sig);
            std::string item;
            while (std::getline(ss, item, '|'))
                if (!item.empty())
                    out.insert(item);
            return out;
        };

        auto sigSet = toSet(signature);
        json best;
        double bestScore = 0;
        for (const auto &id : candidateIds)
        {
            auto rec = store_->get("d:" + id);
            if (!rec || !rec->contains("reply"))
                continue;
            if ((*rec)["reply"].is_string())
            {
                auto r = (*rec)["reply"].get<std::string>();
                auto start = r.find_first_not_of(" \n\r\t");
                if (start == std::string::npos)
                    continue;
            }
            auto recSet = toSet(rec->value("signature", ""));
            double inter = 0;
            for (auto &x : sigSet)
                if (recSet.count(x))
                    inter++;
            double uni = sigSet.size() + recSet.size() - inter;
            double sim = (uni <= 0) ? 0 : inter / uni;
            if (sim < minSim)
                continue;
            double freq = std::log(1 + rec->value("count", 0));
            double score = sim * (1 + 0.15 * freq);
            if (score > bestScore)
            {
                bestScore = score;
                best = *rec;
                best["similarity"] = sim;
                best["score"] = score;
            }
        }
        return best;
    }

    json exportSnapshot(int limit = 512)
    {
        json out{{"dialogs", json::array()}, {"indexes", json::array()}};
        auto dialogs = store_->entries("d:");
        for (auto &kv : dialogs)
        {
            out["dialogs"].push_back({kv.first, kv.second});
            if ((int)out["dialogs"].size() >= limit)
                break;
        }
        auto indexes = store_->entries("i:");
        for (auto &kv : indexes)
        {
            out["indexes"].push_back({kv.first, kv.second});
            if ((int)out["indexes"].size() >= limit)
                break;
        }
        return out;
    }

    void importSnapshot(const json &snapshot)
    {
        for (auto &item : snapshot.value("dialogs", json::array()))
        {
            if (!item.is_array() || item.size() != 2)
                continue;
            store_->put(item[0].get<std::string>(), item[1]);
        }
        for (auto &item : snapshot.value("indexes", json::array()))
        {
            if (!item.is_array() || item.size() != 2)
                continue;
            store_->put(item[0].get<std::string>(), item[1]);
        }
    }

private:
    static std::string scopeKey(const std::string &sessionId)
    {
        if (sessionId.empty())
            return std::string();
        return "s" + sha1Hex(sessionId);
    }

    std::shared_ptr<KeyValueStore> store_;
    int maxItems_;
    int maxPerIndex_;
};

// ------------------ Robots / Study / Shards ------------------

class ControllerPoolBase;
class RedisSynchronizer;

// RobotsDocument 表示语料文档的标准化结构。
// 调用方式：语料收集与分发流程统一使用该对象。
// 实现思路：聚合 id、文本、分词与来源元信息。
// 注意事项：text 可能较大，序列化时需关注开销。
// 注意事项：tokens 可为空，上层需支持延迟分词。
// 注意事项：source/file 字段用于追溯，不用于鉴权。
struct RobotsDocument
{
    std::string id;
    std::string text;
    std::vector<std::string> tokens;
    std::string source;
    std::string file;
};

// RobotsCorpus 管理 robots 语料加载、切分与标准化。
// 调用方式：在训练热身或重建阶段按选项批量调用。
// 实现思路：遍历文本文件并结合词形还原构建文档。
// 注意事项：目录规模较大时建议配合 limit/offset。
// 注意事项：读取异常不会中断全局流程，需看返回结果。
// 注意事项：分块参数会影响后续学习效果。
class RobotsCorpus
{
public:
    RobotsCorpus(fs::path dir,
                 fs::path lemmaCsv,
                 bool lemmaAutoload,
                 std::size_t lemmaMaxBytes,
                 bool lemmaForce,
                 int chunkMinWords,
                 int chunkMaxWords)
        : dir_(std::move(dir)),
          lemmaCsv_(std::move(lemmaCsv)),
          lemmaAutoload_(lemmaAutoload),
          lemmaForce_(lemmaForce),
          lemmaMaxBytes_(lemmaMaxBytes > 0 ? lemmaMaxBytes : 64ull * 1024ull * 1024ull),
          chunkMinWords_(std::max(1, chunkMinWords)),
          chunkMaxWords_(std::max(chunkMinWords_, chunkMaxWords)) {}

    std::vector<std::string> listFiles() const
    {
        std::vector<std::string> files;
        if (dir_.empty() || !fs::exists(dir_))
            return files;
        try
        {
            for (auto &entry : fs::directory_iterator(dir_))
            {
                if (!entry.is_regular_file())
                    continue;
                auto ext = entry.path().extension().string();
                std::string lowerExt;
                lowerExt.reserve(ext.size());
                for (char c : ext)
                    lowerExt.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
                if (lowerExt == ".txt")
                    files.push_back(entry.path().filename().string());
            }
            std::sort(files.begin(), files.end());
        }
        catch (const std::exception &e)
        {
            std::cerr << "[RobotsCorpus] Failed to list files: " << e.what() << std::endl;
        }
        return files;
    }

    std::vector<std::string> normalizeWords(const std::vector<std::string> &words)
    {
        std::vector<std::string> out;
        out.reserve(words.size());
        for (auto &w : words)
        {
            auto lw = lemmatize(w);
            if (!lw.empty())
                out.push_back(lw);
        }
        return out;
    }

    std::vector<RobotsDocument> collect(const json &options)
    {
        std::vector<RobotsDocument> docs;
        auto files = listFiles();
        if (options.contains("files") && options["files"].is_array())
        {
            std::unordered_set<std::string> allow;
            for (auto &v : options["files"])
                if (v.is_string())
                    allow.insert(v.get<std::string>());
            files.erase(std::remove_if(files.begin(), files.end(), [&](const std::string &f)
                                       { return !allow.count(f); }),
                        files.end());
        }
        if (options.value("shuffle", false) && files.size() > 1)
        {
            std::mt19937 rng((unsigned)std::chrono::high_resolution_clock::now().time_since_epoch().count());
            std::shuffle(files.begin(), files.end(), rng);
        }
        int offset = options.value("offset", 0);
        if (offset < 0)
            offset = 0;
        int limit = options.contains("limit") ? options.value("limit", 0) : 0;
        int maxDocs = (limit > 0) ? limit : 0;

        for (const auto &file : files)
        {
            auto content = readFileLimited(dir_ / file, maxArticleSize_);
            if (content.empty())
                continue;
            auto paragraphs = splitParagraphs(content);
            int localIndex = 0;
            for (const auto &paragraph : paragraphs)
            {
                std::string trimmed = paragraph;
                trimmed.erase(0, trimmed.find_first_not_of(" \n\r\t"));
                trimmed.erase(trimmed.find_last_not_of(" \n\r\t") + 1);
                if ((int)trimmed.size() < minParagraphLength_)
                    continue;
                auto units = splitSentences(trimmed);
                for (const auto &unit : units)
                {
                    std::string unitText = unit;
                    unitText.erase(0, unitText.find_first_not_of(" \n\r\t"));
                    unitText.erase(unitText.find_last_not_of(" \n\r\t") + 1);
                    if ((int)unitText.size() < minParagraphLength_)
                        continue;
                    auto tokens = normalizeWords(tokenize(unitText));
                    if (tokens.empty())
                        continue;
                    if (offset > 0)
                    {
                        offset--;
                        continue;
                    }
                    RobotsDocument doc;
                    doc.id = "robots:" + file + "#" + std::to_string(localIndex);
                    doc.file = file;
                    doc.source = "robots:" + file;
                    doc.text = unitText;
                    doc.tokens = std::move(tokens);
                    docs.push_back(std::move(doc));
                    localIndex++;
                    if (maxDocs > 0 && (int)docs.size() >= maxDocs)
                        return docs;
                }
            }
        }
        if (options.value("shuffle", false) && docs.size() > 1)
        {
            std::mt19937 rng((unsigned)std::chrono::high_resolution_clock::now().time_since_epoch().count());
            std::shuffle(docs.begin(), docs.end(), rng);
        }
        return docs;
    }

    std::string lemmatize(const std::string &word)
    {
        ensureLemmaLoaded();
        std::string lower;
        lower.reserve(word.size());
        for (unsigned char c : word)
        {
            if (c < 0x80)
                lower.push_back(static_cast<char>(std::tolower(c)));
            else
                lower.push_back(static_cast<char>(c));
        }
        auto it = lemmaMap_.find(lower);
        return it == lemmaMap_.end() ? lower : it->second;
    }

private:
    fs::path dir_;
    fs::path lemmaCsv_;
    bool lemmaAutoload_{false};
    bool lemmaForce_{false};
    std::size_t lemmaMaxBytes_{64ull * 1024ull * 1024ull};
    int chunkMinWords_{2};
    int chunkMaxWords_{5};
    bool lemmaLoaded_{false};
    std::unordered_map<std::string, std::string> lemmaMap_;
    std::size_t maxArticleSize_{5000000};
    int minParagraphLength_{12};

    void ensureLemmaLoaded()
    {
        if (lemmaLoaded_)
            return;
        lemmaLoaded_ = true;
        if (!lemmaAutoload_ && !lemmaForce_)
            return;
        lemmaMap_ = loadLemmaMap();
    }

    std::unordered_map<std::string, std::string> loadLemmaMap()
    {
        std::unordered_map<std::string, std::string> map;
        if (lemmaCsv_.empty() || !fs::exists(lemmaCsv_))
            return map;
        try
        {
            if (!lemmaForce_)
            {
                try
                {
                    auto st = fs::file_size(lemmaCsv_);
                    if (st > lemmaMaxBytes_)
                    {
                        std::cerr << "[RobotsCorpus] lemma.csv too large (" << (st / 1024 / 1024) << "MB), skip autoload. Set AI_LEMMA_FORCE=true or increase AI_LEMMA_MAX_MB to load." << std::endl;
                        return map;
                    }
                }
                catch (...)
                {
                }
            }
            std::ifstream in(lemmaCsv_, std::ios::binary);
            std::string line;
            while (std::getline(in, line))
            {
                if (line.empty())
                    continue;
                auto cols = splitCsvLine(line);
                if (cols.empty())
                    continue;
                std::string base = toLower(cols[0]);
                if (base.empty())
                    continue;
                map[base] = base;
                for (auto &c : cols)
                {
                    auto token = toLower(c);
                    if (!token.empty())
                        map[token] = base;
                }
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << "[RobotsCorpus] Failed to load lemma map: " << e.what() << std::endl;
        }
        return map;
    }

    static std::string toLower(const std::string &s)
    {
        std::string out;
        out.reserve(s.size());
        for (unsigned char c : s)
        {
            if (c < 0x80)
                out.push_back(static_cast<char>(std::tolower(c)));
            else
                out.push_back(static_cast<char>(c));
        }
        return out;
    }

    static std::vector<std::string> splitCsvLine(const std::string &line)
    {
        std::vector<std::string> cols;
        std::string cur;
        bool inQuotes = false;
        for (size_t i = 0; i < line.size(); i++)
        {
            char c = line[i];
            if (c == '"')
            {
                if (inQuotes && i + 1 < line.size() && line[i + 1] == '"')
                {
                    cur.push_back('"');
                    i++;
                }
                else
                {
                    inQuotes = !inQuotes;
                }
            }
            else if (c == ',' && !inQuotes)
            {
                cols.push_back(cur);
                cur.clear();
            }
            else
            {
                cur.push_back(c);
            }
        }
        cols.push_back(cur);
        return cols;
    }

    static std::string readFileLimited(const fs::path &file, std::size_t maxBytes)
    {
        try
        {
            std::ifstream in(file, std::ios::binary);
            if (!in)
                return {};
            std::string out;
            out.reserve(4096);
            char buf[4096];
            std::size_t total = 0;
            while (in && total < maxBytes)
            {
                in.read(buf, sizeof(buf));
                auto got = (std::size_t)in.gcount();
                if (!got)
                    break;
                if (total + got > maxBytes)
                    got = maxBytes - total;
                out.append(buf, got);
                total += got;
            }
            return out;
        }
        catch (...)
        {
            return {};
        }
    }

    static std::vector<std::string> splitParagraphs(const std::string &content)
    {
        std::vector<std::string> out;
        std::regex re(R"(\r?\n\s*\r?\n)");
        std::sregex_token_iterator it(content.begin(), content.end(), re, -1);
        std::sregex_token_iterator end;
        for (; it != end; ++it)
        {
            std::string s = it->str();
            s.erase(0, s.find_first_not_of(" \n\r\t"));
            s.erase(s.find_last_not_of(" \n\r\t") + 1);
            if (!s.empty())
                out.push_back(s);
        }
        return out;
    }
};

class SessionManager
{
public:
    SessionManager(std::shared_ptr<KeyValueStore> store, int idleMs = 10 * 60 * 1000, int maxSessions = 200)
        : store_(std::move(store)), idleMs_(idleMs), maxSessions_(maxSessions) {}

    std::string ensure(const std::string &sessionId)
    {
        std::lock_guard<std::mutex> lock(mu_);
        loadFromStoreLocked();
        cleanupExpiredLocked(nowMs());

        if (!sessionId.empty())
        {
            std::string sid = sessionId;
            auto it = active_.find(sid);
            if (it != active_.end())
            {
                auto &data = it->second;
                data["lastActivity"] = nowMs();
                data["count"] = data.value("count", 0) + 1;
                saveLocked(sid, data);
                if ((int)active_.size() > maxSessions_)
                    truncateLocked();
                return sid;
            }
            auto stored = store_->get("session:" + sid);
            if (stored && stored->is_object())
            {
                json data = *stored;
                data["id"] = data.value("id", sid);
                data["lastActivity"] = nowMs();
                data["count"] = data.value("count", 0) + 1;
                active_[sid] = data;
                if (std::find(order_.begin(), order_.end(), sid) == order_.end())
                    order_.push_back(sid);
                saveLocked(sid, data);
                if ((int)active_.size() > maxSessions_)
                    truncateLocked();
                return sid;
            }
        }

        std::string id = newIdLocked();
        json data{{"id", id}, {"createdAt", nowMs()}, {"lastActivity", nowMs()}, {"count", 1}, {"meta", json::object()}};
        active_[id] = data;
        order_.push_back(id);
        saveLocked(id, data);
        if ((int)active_.size() > maxSessions_)
            truncateLocked();
        return id;
    }

    json exportSessions() const
    {
        std::lock_guard<std::mutex> lock(mu_);
        json out = json::array();
        for (auto &id : order_)
        {
            auto it = active_.find(id);
            if (it != active_.end())
                out.push_back(it->second);
        }
        return out;
    }

    void importSessions(const json &list)
    {
        std::lock_guard<std::mutex> lock(mu_);
        active_.clear();
        order_.clear();
        loaded_ = true;
        if (!list.is_array())
            return;
        for (auto &item : list)
        {
            if (!item.is_object())
                continue;
            std::string id = item.value("id", "");
            if (id.empty())
                continue;
            active_[id] = item;
            order_.push_back(id);
            saveLocked(id, item);
        }
        cleanupExpiredLocked(nowMs());
        if ((int)active_.size() > maxSessions_)
            truncateLocked();
    }

private:
    std::shared_ptr<KeyValueStore> store_;
    int idleMs_{10 * 60 * 1000};
    int maxSessions_{200};
    std::unordered_map<std::string, json> active_;
    std::vector<std::string> order_;
    bool loaded_{false};
    mutable std::mutex mu_;

    static int64_t nowMs()
    {
        return (int64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::system_clock::now().time_since_epoch())
            .count();
    }

    void saveLocked(const std::string &sid, const json &data)
    {
        store_->put("session:" + sid, data);
    }

    std::string newIdLocked()
    {
        for (int i = 0; i < 8; i++)
        {
            std::string candidate = randomSessionId();
            if (!active_.count(candidate))
                return candidate;
        }
        return randomHex(16);
    }

    void loadFromStoreLocked()
    {
        if (loaded_)
            return;
        loaded_ = true;
        auto items = store_->entries("session:");
        for (auto &kv : items)
        {
            if (!kv.second.is_object())
                continue;
            std::string sid = kv.second.value("id", "");
            if (sid.empty())
            {
                if (kv.first.rfind("session:", 0) == 0)
                    sid = kv.first.substr(8);
            }
            if (sid.empty())
                continue;
            active_[sid] = kv.second;
            active_[sid]["id"] = sid;
            order_.push_back(sid);
        }
    }

    void cleanupExpiredLocked(int64_t now)
    {
        if (idleMs_ <= 0)
            return;
        std::unordered_set<std::string> keep;
        for (auto it = active_.begin(); it != active_.end();)
        {
            const auto &obj = it->second;
            int64_t last = obj.value("lastActivity", obj.value("createdAt", 0ll));
            if (now - last > idleMs_)
            {
                store_->del("session:" + it->first);
                it = active_.erase(it);
            }
            else
            {
                keep.insert(it->first);
                ++it;
            }
        }
        if (!order_.empty())
        {
            std::vector<std::string> next;
            next.reserve(order_.size());
            for (const auto &id : order_)
            {
                if (keep.count(id))
                    next.push_back(id);
            }
            order_.swap(next);
        }
    }

    void truncateLocked()
    {
        std::vector<json> items;
        items.reserve(active_.size());
        for (auto &kv : active_)
            items.push_back(kv.second);
        std::sort(items.begin(), items.end(), [](const json &a, const json &b)
                  { return a.value("lastActivity", 0) > b.value("lastActivity", 0); });
        if ((int)items.size() <= maxSessions_)
            return;
        std::unordered_set<std::string> keep;
        for (int i = 0; i < maxSessions_ && i < (int)items.size(); i++)
            keep.insert(items[i].value("id", ""));
        for (auto it = active_.begin(); it != active_.end();)
        {
            if (!keep.count(it->first))
            {
                store_->del("session:" + it->first);
                it = active_.erase(it);
            }
            else
            {
                ++it;
            }
        }
        if (!order_.empty())
        {
            std::vector<std::string> next;
            next.reserve(order_.size());
            for (auto &id : order_)
            {
                if (keep.count(id))
                    next.push_back(id);
            }
            order_.swap(next);
        }
    }
};

struct LemmaCacheEntry
{
    fs::file_time_type mtime;
    std::unordered_map<std::string, std::string> map;
};

static std::mutex gLemmaCacheMu;
static std::unordered_map<std::string, LemmaCacheEntry> gLemmaCache;

static std::string trimString(const std::string &s)
{
    auto start = s.find_first_not_of(" \n\r\t");
    if (start == std::string::npos)
        return std::string();
    auto end = s.find_last_not_of(" \n\r\t");
    return s.substr(start, end - start + 1);
}

static std::string lowerAscii(const std::string &s)
{
    std::string out;
    out.reserve(s.size());
    for (unsigned char c : s)
    {
        if (c < 0x80)
            out.push_back(static_cast<char>(std::tolower(c)));
        else
            out.push_back(static_cast<char>(c));
    }
    return out;
}

static std::vector<std::string> splitCsvLineSimple(const std::string &line)
{
    std::vector<std::string> cols;
    std::string cur;
    bool inQuotes = false;
    for (size_t i = 0; i < line.size(); i++)
    {
        char c = line[i];
        if (c == '"')
        {
            if (inQuotes && i + 1 < line.size() && line[i + 1] == '"')
            {
                cur.push_back('"');
                i++;
            }
            else
            {
                inQuotes = !inQuotes;
            }
        }
        else if (c == ',' && !inQuotes)
        {
            cols.push_back(trimString(cur));
            cur.clear();
        }
        else
        {
            cur.push_back(c);
        }
    }
    cols.push_back(trimString(cur));
    return cols;
}

static std::unordered_map<std::string, std::string> loadLemmaMapCached(const std::string &lemmaCsvPath)
{
    std::unordered_map<std::string, std::string> map;
    if (lemmaCsvPath.empty())
        return map;
    fs::path file(lemmaCsvPath);
    try
    {
        if (!fs::exists(file))
            return map;
        auto mtime = fs::last_write_time(file);
        {
            std::lock_guard<std::mutex> lock(gLemmaCacheMu);
            auto it = gLemmaCache.find(file.string());
            if (it != gLemmaCache.end() && it->second.mtime >= mtime)
            {
                return it->second.map;
            }
        }
        std::ifstream in(file, std::ios::binary);
        if (!in)
            return map;
        std::string line;
        while (std::getline(in, line))
        {
            if (line.empty())
                continue;
            auto cols = splitCsvLineSimple(line);
            if (cols.empty())
                continue;
            std::string base = lowerAscii(cols[0]);
            base = trimString(base);
            if (base.empty())
                continue;
            map[base] = base;
            for (auto &c : cols)
            {
                auto token = trimString(lowerAscii(c));
                if (!token.empty())
                    map[token] = base;
            }
        }
        {
            std::lock_guard<std::mutex> lock(gLemmaCacheMu);
            gLemmaCache[file.string()] = LemmaCacheEntry{mtime, map};
        }
    }
    catch (...)
    {
        return map;
    }
    return map;
}

class LemmaWorkerPool
{
public:
    explicit LemmaWorkerPool(int workers)
    {
        int count = std::max(1, workers);
        running_ = true;
        threads_.reserve(count);
        for (int i = 0; i < count; i++)
        {
            threads_.emplace_back([this]()
                                  { workerLoop(); });
        }
    }

    ~LemmaWorkerPool()
    {
        stop();
    }

    std::vector<std::string> execBatchLemmatize(const std::vector<std::string> &tokens, const std::string &lemmaCsvPath)
    {
        if (tokens.empty())
            return {};
        auto task = std::make_shared<Task>();
        task->tokens = tokens;
        task->lemmaCsvPath = lemmaCsvPath;
        auto fut = task->promise.get_future();
        {
            std::lock_guard<std::mutex> lock(mu_);
            queue_.push(task);
        }
        cv_.notify_one();
        return fut.get();
    }

    void stop()
    {
        if (!running_)
            return;
        running_ = false;
        cv_.notify_all();
        for (auto &t : threads_)
        {
            if (t.joinable())
                t.join();
        }
        threads_.clear();
    }

private:
    struct Task
    {
        std::vector<std::string> tokens;
        std::string lemmaCsvPath;
        std::promise<std::vector<std::string>> promise;
    };

    std::atomic<bool> running_{false};
    std::vector<std::thread> threads_;
    std::queue<std::shared_ptr<Task>> queue_;
    std::mutex mu_;
    std::condition_variable cv_;

    void workerLoop()
    {
        while (running_)
        {
            std::shared_ptr<Task> task;
            {
                std::unique_lock<std::mutex> lock(mu_);
                cv_.wait(lock, [&]()
                         { return !running_ || !queue_.empty(); });
                if (!running_ && queue_.empty())
                    return;
                task = queue_.front();
                queue_.pop();
            }
            try
            {
                auto lemmaMap = loadLemmaMapCached(task->lemmaCsvPath);
                std::vector<std::string> out;
                out.reserve(task->tokens.size());
                for (auto &w : task->tokens)
                {
                    std::string normalized = trimString(lowerAscii(w));
                    if (normalized.empty())
                    {
                        out.push_back(std::string());
                        continue;
                    }
                    auto it = lemmaMap.find(normalized);
                    if (it != lemmaMap.end())
                        out.push_back(it->second);
                    else
                        out.push_back(normalized);
                }
                task->promise.set_value(out);
            }
            catch (...)
            {
                try
                {
                    task->promise.set_value({});
                }
                catch (...)
                {
                }
            }
        }
    }
};

class StudyEngine
    : public IStudyEngine
{
public:
    StudyEngine(std::shared_ptr<ControllerPoolBase> pool, std::shared_ptr<IRedisSynchronizer> redis);

    void start() override;
    void enqueueDocument(const json &doc) override;
    json status() const override;
    bool running() const override { return running_; }

private:
    void tick();
    std::shared_ptr<LemmaWorkerPool> ensureWorkerPool();

    std::shared_ptr<ControllerPoolBase> pool_;
    std::shared_ptr<IRedisSynchronizer> redis_;
    std::atomic<bool> running_{false};
    std::thread worker_;
    mutable std::mutex mu_;
    std::deque<json> queue_;
    json metrics_ = json{{"enqueued", 0}, {"processed", 0}, {"lastTickAt", 0}, {"lastError", nullptr}};
    mutable std::mutex workerMu_;
    std::shared_ptr<LemmaWorkerPool> workerPool_;
};

// ------------------ Runtime ------------------

class RuntimeState : public RuntimeBase
{
public:
    RuntimeState(std::shared_ptr<KeyValueStore> kvmStore, std::shared_ptr<KeyValueStore> memeStore, std::shared_ptr<KeyValueStore> sessionStore, const Config &config)
        : config_(config), kvm_(kvmStore, config.kvmCacheMaxEntries), graph_(memeStore), sessions_(sessionStore), pattern_(&kvm_, &tensor_)
    {
        surfaceStore_ = std::make_shared<NamespacedStore>(sessionStore, "surface");
        dialogStore_ = std::make_shared<NamespacedStore>(sessionStore, "dialog");
        surfaceLexicon_ = std::make_shared<MemeSurfaceLexicon>(surfaceStore_);
        dialogMemory_ = std::make_shared<DialogMemory>(dialogStore_);
        sessionManager_ = std::make_shared<SessionManager>(sessionStore);
        params_ = MODEL_DEFAULTS;
        searchConfig_.enabled = true;
        if (!config_.searchEndpoint.empty())
        {
            searchConfig_.active = config_.searchEndpoint;
            searchConfig_.endpoints.push_back(config_.searchEndpoint);
        }
        if (!searchConfig_.active.empty() && std::find(searchConfig_.endpoints.begin(), searchConfig_.endpoints.end(), searchConfig_.active) == searchConfig_.endpoints.end())
        {
            searchConfig_.endpoints.push_back(searchConfig_.active);
        }
        researcher_ = std::make_shared<OnlineResearcher>(this, searchConfig_.active);
        researcher_->setEnabled(searchConfig_.enabled);
        addons_ = addon::createDefaultAddons();
    }

    json cloneParams() const override
    {
        json j;
        j["decayFactor"] = params_.decayFactor;
        j["maxMemeWords"] = params_.maxMemeWords;
        j["minOverlapThreshold"] = params_.minOverlapThreshold;
        j["memeNgramMin"] = params_.memeNgramMin;
        j["memeNgramMax"] = params_.memeNgramMax;
        j["maliciousThreshold"] = params_.maliciousThreshold;
        j["learningIterations"] = params_.learningIterations;
        j["iteration"] = params_.iteration;
        j["threshold"] = params_.threshold;
        j["decay"] = params_.decay;
        j["decayK"] = params_.decayK;
        j["maxLen"] = params_.maxLen;
        j["edgeWeight"] = params_.edgeWeight;
        j["activationType"] = params_.activationType;
        j["transferType"] = params_.transferType;
        j["activationCustom"] = params_.activationCustom;
        j["transferCustom"] = params_.transferCustom;
        j["mappingDepth"] = params_.mappingDepth;
        j["reflectionTopMemes"] = params_.reflectionTopMemes;
        j["reflectionTopWords"] = params_.reflectionTopWords;
        j["reflectionMinScore"] = params_.reflectionMinScore;
        return j;
    }

    json listAddons() const
    {
        return addons_ ? addons_->listAddons() : json::array();
    }

    bool addAddon(const std::string &type, const std::string &name, std::string &error)
    {
        if (!addons_)
        {
            error = "addon manager unavailable";
            return false;
        }
        return addons_->addBuiltin(type, name, &error);
    }

    bool addAddon(const std::string &path, std::string &error)
    {
        if (!addons_)
        {
            error = "addon manager unavailable";
            return false;
        }
        return addons_->loadLibrary(path, &error);
    }

    bool removeAddon(const std::string &name, std::string &error)
    {
        if (!addons_)
        {
            error = "addon manager unavailable";
            return false;
        }
        return addons_->removeAddon(name, &error);
    }

    addon::AddonResult runAddonTool(const std::string &text, const json &payload)
    {
        if (!addons_)
            return addon::AddonResult{};
        addon::setAddonOnlineLookupHandler([this](const json &input, const json &options)
                                           { return this->onlineLookup(input, options); });
        struct LookupGuard
        {
            ~LookupGuard() { addon::clearAddonOnlineLookupHandler(); }
        } guard;
        auto result = addons_->run(text, payload);
        return result;
    }

    void setParams(const json &patch) override
    {
        if (patch.contains("decayFactor"))
            params_.decayFactor = patch["decayFactor"].get<double>();
        if (patch.contains("maxMemeWords"))
            params_.maxMemeWords = patch["maxMemeWords"].get<int>();
        if (patch.contains("minOverlapThreshold"))
            params_.minOverlapThreshold = patch["minOverlapThreshold"].get<int>();
        if (patch.contains("memeNgramMin"))
            params_.memeNgramMin = patch["memeNgramMin"].get<int>();
        if (patch.contains("memeNgramMax"))
            params_.memeNgramMax = patch["memeNgramMax"].get<int>();
        if (patch.contains("maliciousThreshold"))
            params_.maliciousThreshold = patch["maliciousThreshold"].get<double>();
        if (patch.contains("learningIterations"))
            params_.learningIterations = patch["learningIterations"].get<int>();
        if (patch.contains("iteration"))
            params_.iteration = patch["iteration"].get<int>();
        if (patch.contains("threshold"))
            params_.threshold = patch["threshold"].get<double>();
        if (patch.contains("decay"))
            params_.decay = patch["decay"].get<double>();
        if (patch.contains("decayK"))
            params_.decayK = patch["decayK"].get<double>();
        if (patch.contains("maxLen"))
            params_.maxLen = patch["maxLen"].get<int>();
        if (patch.contains("edgeWeight"))
            params_.edgeWeight = patch["edgeWeight"].get<double>();
        if (patch.contains("activationType"))
            params_.activationType = patch["activationType"].get<std::string>();
        if (patch.contains("transferType"))
            params_.transferType = patch["transferType"].get<std::string>();
        if (patch.contains("activationCustom"))
            params_.activationCustom = patch["activationCustom"].get<std::string>();
        if (patch.contains("transferCustom"))
            params_.transferCustom = patch["transferCustom"].get<std::string>();
        if (patch.contains("mappingDepth"))
            params_.mappingDepth = patch["mappingDepth"].get<int>();
        if (patch.contains("reflectionTopMemes"))
            params_.reflectionTopMemes = patch["reflectionTopMemes"].get<int>();
        if (patch.contains("reflectionTopWords"))
            params_.reflectionTopWords = patch["reflectionTopWords"].get<int>();
        if (patch.contains("reflectionMinScore"))
            params_.reflectionMinScore = patch["reflectionMinScore"].get<double>();
    }

    struct SeedMapping
    {
        std::unordered_map<std::string, double> seeds;
        std::vector<std::string> order;
    };

    SeedMapping mapWordsToMemesOrdered(const std::vector<std::string> &words)
    {
        SeedMapping out;
        auto normalizeToken = [](std::string s)
        {
            s = trimCopy(s);
            if (s.empty())
                return s;
            std::transform(s.begin(), s.end(), s.begin(), [](unsigned char ch)
                           { return (char)std::tolower(ch); });
            while (!s.empty())
            {
                unsigned char c = (unsigned char)s.front();
                if (std::isalnum(c) || c == '_' || c == '-' || c >= 0x80)
                    break;
                s.erase(s.begin());
            }
            while (!s.empty())
            {
                unsigned char c = (unsigned char)s.back();
                if (std::isalnum(c) || c == '_' || c == '-' || c >= 0x80)
                    break;
                s.pop_back();
            }
            return s;
        };

        std::vector<std::string> rawTokens;
        rawTokens.reserve(words.size());
        for (auto &w : words)
        {
            std::string t = normalizeToken(w);
            if (!t.empty())
                rawTokens.push_back(t);
        }

        const auto &stop = getStopWords();
        std::vector<std::string> tokens;
        tokens.reserve(rawTokens.size());
        for (auto &t : rawTokens)
        {
            bool hasNonAscii = false;
            for (unsigned char c : t)
            {
                if (c >= 0x80)
                {
                    hasNonAscii = true;
                    break;
                }
            }
            if (!hasNonAscii)
            {
                if (t.size() <= 1)
                    continue;
                if (stop.count(t))
                    continue;
            }
            tokens.push_back(t);
        }
        if (tokens.empty())
            tokens = rawTokens;

        std::unordered_map<std::string, int> tokenFreq;
        for (const auto &t : tokens)
            tokenFreq[t]++;

        auto tokenWeight = [&](const std::string &token, size_t idx)
        {
            double posDecay = 1.0 / (1.0 + 0.03 * (double)idx);
            double freqBoost = 1.0 + 0.15 * std::log1p((double)tokenFreq[token]);
            return posDecay * freqBoost;
        };

        auto addSeed = [&](const std::string &memeId, double delta)
        {
            if (memeId.empty())
                return;
            auto it = out.seeds.find(memeId);
            if (it == out.seeds.end())
            {
                out.seeds[memeId] = delta;
                out.order.push_back(memeId);
            }
            else
            {
                it->second += delta;
            }
        };

        for (size_t wi = 0; wi < tokens.size(); ++wi)
        {
            const auto &word = tokens[wi];
            auto memes = kvm_.getWordMemeSet(word);
            if (memes.empty())
            {
                std::string memeId = "meme_" + sha1Hex(word);
                graph_.ensureNode(memeId);
                kvm_.link(word, memeId);
                addSeed(memeId, tokenWeight(word, wi));
            }
            else
            {
                for (auto &mid : memes)
                    addSeed(mid, tokenWeight(word, wi));
            }
        }

        int nMin = std::max(2, params_.memeNgramMin);
        int nMax = std::max(nMin, params_.memeNgramMax);
        int minOverlap = std::max(1, params_.minOverlapThreshold);
        int maxWordSet = std::max(4, params_.maxMemeWords);
        int units = 0;
        int maxUnits = 128;
        std::unordered_set<std::string> seenNgram;

        auto resolveOrCreate = [&](const std::vector<std::string> &tokenSet) -> std::string
        {
            std::vector<std::string> uniq;
            std::unordered_set<std::string> seen;
            for (auto &t : tokenSet)
            {
                if (t.empty())
                    continue;
                if (seen.insert(t).second)
                    uniq.push_back(t);
            }
            if ((int)uniq.size() > maxWordSet)
                uniq.resize(maxWordSet);
            if (uniq.size() <= 1)
                return uniq.empty() ? "" : ("meme_" + sha1Hex(uniq[0]));

            std::unordered_map<std::string, int> counts;
            for (auto &w : uniq)
            {
                for (auto &mid : kvm_.getWordMemeSet(w))
                    counts[mid]++;
            }
            std::string best;
            int bestOverlap = 0;
            for (auto &kv : counts)
            {
                if (kv.second > bestOverlap)
                {
                    bestOverlap = kv.second;
                    best = kv.first;
                }
            }
            if (!best.empty() && bestOverlap >= minOverlap)
            {
                for (auto &w : uniq)
                    kvm_.link(w, best);
                graph_.ensureNode(best);
                return best;
            }
            std::vector<std::string> sorted = uniq;
            std::sort(sorted.begin(), sorted.end());
            std::string memeId = "meme_p_" + sha1Hex([&]()
                                                     {
                std::string join; for (size_t i = 0; i < sorted.size(); i++) { if (i) join += "|"; join += sorted[i]; } return join; }());
            graph_.ensureNode(memeId);
            for (auto &w : sorted)
                kvm_.link(w, memeId);
            return memeId;
        };

        for (size_t i = 0; i < tokens.size() && units < maxUnits; i++)
        {
            for (int n = nMin; n <= nMax && units < maxUnits; n++)
            {
                if (i + n > tokens.size())
                    break;
                std::vector<std::string> gram(tokens.begin() + i, tokens.begin() + i + n);
                std::string gramKey;
                gramKey.reserve(n * 8);
                for (int gi = 0; gi < n; gi++)
                {
                    if (gi)
                        gramKey.push_back('\x1f');
                    gramKey += gram[(size_t)gi];
                }
                if (!seenNgram.insert(gramKey).second)
                    continue;
                std::string memeId = resolveOrCreate(gram);
                if (memeId.empty())
                    continue;
                std::unordered_set<std::string> uniq(gram.begin(), gram.end());
                double uniqRatio = (double)uniq.size() / std::max(1, n);
                double w = (1.0 + 0.45 * (n - 1)) * uniqRatio;
                w *= 1.0 / (1.0 + 0.02 * (double)i);
                addSeed(memeId, w);
                units++;
            }
        }

        return out;
    }

    std::unordered_map<std::string, double> mapWordsToMemes(const std::vector<std::string> &words) override
    {
        return mapWordsToMemesOrdered(words).seeds;
    }

    json processInput(const json &payload) override
    {
        auto started = std::chrono::steady_clock::now();
        std::function<std::string(const json &)> jsStringForJoin;
        std::function<std::string(const json &)> jsStringValue;
        jsStringForJoin = [&](const json &v) -> std::string
        {
            if (v.is_null())
                return "";
            if (v.is_string())
                return v.get<std::string>();
            if (v.is_boolean())
                return v.get<bool>() ? "true" : "false";
            if (v.is_number())
                return v.dump();
            if (v.is_array())
            {
                std::string out;
                bool first = true;
                for (auto &item : v)
                {
                    if (!first)
                        out += ",";
                    first = false;
                    out += jsStringForJoin(item);
                }
                return out;
            }
            return "[object Object]";
        };
        jsStringValue = [&](const json &v) -> std::string
        {
            if (v.is_null())
                return "null";
            if (v.is_string())
                return v.get<std::string>();
            if (v.is_boolean())
                return v.get<bool>() ? "true" : "false";
            if (v.is_number())
                return v.dump();
            if (v.is_array())
                return jsStringForJoin(v);
            return "[object Object]";
        };
        auto jsStringOrEmpty = [&](const json &v) -> std::string
        {
            if (v.is_null())
                return "";
            if (v.is_boolean())
                return v.get<bool>() ? "true" : "";
            if (v.is_number())
            {
                double n = v.get<double>();
                if (n == 0.0)
                    return "";
                return v.dump();
            }
            if (v.is_string())
                return v.get<std::string>();
            if (v.is_array())
            {
                std::string out;
                bool first = true;
                for (auto &item : v)
                {
                    if (!first)
                        out += ",";
                    first = false;
                    out += jsStringForJoin(item);
                }
                return out;
            }
            return "[object Object]";
        };
        json textVal = json::object();
        if (payload.contains("text") && !payload["text"].is_null())
            textVal = payload["text"];
        else if (payload.contains("message") && !payload["message"].is_null())
            textVal = payload["message"];
        std::string text = jsStringOrEmpty(textVal);
        auto tokensRaw = payload.value("tokens", payload.value("words", payload.value("vocab", json::array())));
        std::vector<std::string> words;
        if (tokensRaw.is_array() && !tokensRaw.empty())
        {
            for (auto &w : tokensRaw)
            {
                words.push_back(jsStringValue(w));
            }
        }
        else
        {
            words = tokenize(text);
        }
        auto jsTruthy = [&](const json &v) -> bool
        {
            if (v.is_null())
                return false;
            if (v.is_boolean())
                return v.get<bool>();
            if (v.is_number())
            {
                double n = v.get<double>();
                return std::isfinite(n) && n != 0.0;
            }
            if (v.is_string())
                return !v.get<std::string>().empty();
            return true;
        };
        std::string sessionId;
        if (payload.contains("sessionId") && jsTruthy(payload["sessionId"]))
        {
            sessionId = jsStringValue(payload["sessionId"]);
        }
        if (sessionManager_)
            sessionId = sessionManager_->ensure(sessionId);
        if (sessionId.empty())
            sessionId = randomSessionId();

        json budget = payload.contains("budget") && payload["budget"].is_object() ? payload["budget"] : json::object();
        auto jsToNumber = [&](const json &v, bool &ok) -> double
        {
            ok = true;
            if (v.is_null())
            {
                ok = false;
                return std::numeric_limits<double>::quiet_NaN();
            }
            if (v.is_number())
                return v.get<double>();
            if (v.is_boolean())
                return v.get<bool>() ? 1.0 : 0.0;
            if (v.is_string())
            {
                std::string s = v.get<std::string>();
                auto start = s.find_first_not_of(" \n\r\t");
                if (start == std::string::npos)
                    return 0.0;
                auto end = s.find_last_not_of(" \n\r\t");
                s = s.substr(start, end - start + 1);
                if (s.empty())
                    return 0.0;
                try
                {
                    return std::stod(s);
                }
                catch (...)
                {
                    ok = false;
                    return std::numeric_limits<double>::quiet_NaN();
                }
            }
            ok = false;
            return std::numeric_limits<double>::quiet_NaN();
        };
        auto pickValue = [&](const json &obj, const std::vector<std::string> &keys) -> const json *
        {
            for (auto &k : keys)
            {
                if (obj.contains(k) && !obj[k].is_null())
                    return &obj[k];
            }
            return nullptr;
        };
        auto numberOrHard = [&](const json *v, double paramFallback, double hardFallback) -> double
        {
            double n = paramFallback;
            if (v)
            {
                bool ok = false;
                n = jsToNumber(*v, ok);
            }
            if (!std::isfinite(n) || n == 0.0)
                return hardFallback;
            return n;
        };
        auto numberFinite = [&](const json *v, double fallback) -> double
        {
            if (!v)
                return fallback;
            bool ok = false;
            double n = jsToNumber(*v, ok);
            return std::isfinite(n) ? n : fallback;
        };

        int depth = std::max(1, (int)numberOrHard(pickValue(budget, {"mappingDepth", "depth"}), params_.mappingDepth ? params_.mappingDepth : 1, 1));
        int topMemesK = std::max(3, (int)numberOrHard(pickValue(budget, {"reflectionTopMemes", "topMemes"}), params_.reflectionTopMemes ? params_.reflectionTopMemes : 18, 18));
        int topWordsK = std::max(3, (int)numberOrHard(pickValue(budget, {"reflectionTopWords", "topWords"}), params_.reflectionTopWords ? params_.reflectionTopWords : 24, 24));
        double minScoreRaw = numberFinite(pickValue(budget, {"reflectionMinScore", "minScore"}), params_.reflectionMinScore);
        double minScore = std::isfinite(minScoreRaw) ? minScoreRaw : 1e-6;
        int iteration = std::max(1, (int)numberOrHard(pickValue(budget, {"iteration"}), params_.iteration ? params_.iteration : 5, 5));
        int radius = std::max(1, std::min(6, (int)numberOrHard(pickValue(budget, {"radius", "windowRadius"}), 2, 2)));

        auto mapping = mapWordsToMemesOrdered(words);
        auto seeds = mapping.seeds;
        auto seedOrder = mapping.order;
        PropagationResult result;
        if (depth > 1)
        {
            for (int hop = 1; hop < depth; hop++)
            {
                result = runPropagation(seeds, json{{"iteration", iteration}, {"radius", radius}, {"seedOrder", seedOrder}});
                auto topMemes = pickTopActivatedMemes(result, seeds, topMemesK, minScore);
                std::unordered_map<std::string, double> wordScore;
                for (auto &m : topMemes)
                {
                    auto linked = kvm_.getMemeWords(m.memeId);
                    for (auto &w : linked)
                    {
                        if (w.empty())
                            continue;
                        auto prev = wordScore.count(w) ? wordScore[w] : 0.0;
                        wordScore[w] = std::max(prev, m.score);
                    }
                }
                std::vector<std::pair<std::string, double>> sorted(wordScore.begin(), wordScore.end());
                std::sort(sorted.begin(), sorted.end(), [](auto &a, auto &b)
                          { return a.second > b.second; });
                std::vector<std::string> expanded;
                for (size_t i = 0; i < sorted.size() && (int)expanded.size() < topWordsK; i++)
                    expanded.push_back(sorted[i].first);
                std::vector<std::string> merged;
                merged.reserve(std::min<size_t>(64, words.size()) + expanded.size());
                for (size_t i = 0; i < words.size() && i < 64; i++)
                    merged.push_back(words[i]);
                for (auto &w : expanded)
                    merged.push_back(w);
                std::unordered_set<std::string> dedup;
                std::vector<std::string> final;
                for (auto &w : merged)
                    if (!w.empty() && !dedup.count(w))
                    {
                        dedup.insert(w);
                        final.push_back(w);
                    }
                auto nextMapping = mapWordsToMemesOrdered(final);
                seeds = nextMapping.seeds;
                seedOrder = nextMapping.order;
            }
        }
        result = runPropagation(seeds, json{{"iteration", iteration}, {"radius", radius}, {"seedOrder", seedOrder}});
        std::string reply = composeReply(sessionId, result, words, seeds);
        auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - started).count();
        metrics_["requests"] = metrics_.value("requests", 0) + 1;
        metrics_["lastLatency"] = latency;
        metrics_["updatedAt"] = (int64_t)std::chrono::system_clock::now().time_since_epoch().count();

        json out;
        out["reply"] = reply;
        out["seeds"] = json::array();
        for (auto &id : seedOrder)
        {
            auto it = seeds.find(id);
            if (it != seeds.end())
                out["seeds"].push_back({id, it->second});
        }
        out["activation"] = result.activation;
        out["memes"] = result.windowInfo.ids;
        out["sessionId"] = sessionId;
        out["latency"] = latency;
        out["params"] = cloneParams();
        return out;
    }

    json metrics() const { return metrics_; }

    json toSnapshot() override
    {
        json out;
        out["params"] = cloneParams();
        out["graph"] = graph_.exportSnapshot();
        if (sessionManager_)
            out["sessions"] = sessionManager_->exportSessions();
        out["kvm"] = kvm_.exportEntries();
        out["surface"] = surfaceLexicon_->exportSnapshot(512);
        out["dialog"] = dialogMemory_->exportSnapshot(512);
        return out;
    }

    void fromSnapshot(const json &snapshot) override
    {
        if (snapshot.is_null())
            return;
        if (snapshot.contains("params"))
            setParams(snapshot["params"]);
        graph_.importSnapshot(snapshot.value("graph", json::object()));
        if (snapshot.contains("sessions") && sessionManager_)
            sessionManager_->importSessions(snapshot["sessions"]);
        if (snapshot.contains("kvm"))
        {
            for (auto &kv : snapshot["kvm"]["words"].items())
            {
                for (auto &m : kv.value())
                    kvm_.link(kv.key(), m.get<std::string>());
            }
        }
        if (snapshot.contains("surface"))
            surfaceLexicon_->importSnapshot(snapshot["surface"]);
        if (snapshot.contains("dialog"))
            dialogMemory_->importSnapshot(snapshot["dialog"]);
    }

    MemeGraph &graph() { return graph_; }
    KVMStore &kvm() { return kvm_; }
    MemeSurfaceLexicon &surfaceLexicon() { return *surfaceLexicon_; }
    DialogMemory &dialogMemory() { return *dialogMemory_; }
    const Config &config() const override { return config_; }

    std::vector<std::string> getMemeWords(const std::string &memeId) override
    {
        return kvm_.getMemeWords(memeId);
    }

    void graphLink(const std::string &fromId, const std::string &toId, double weight, int direction) override
    {
        graph_.link(fromId, toId, weight, direction);
    }

    json getSearchConfig() const override { return searchConfig_.toJson(); }

    json setSearchConfig(const json &patch) override
    {
        auto trim = [](std::string s)
        {
            auto start = s.find_first_not_of(" \n\r\t");
            if (start == std::string::npos)
                return std::string();
            auto end = s.find_last_not_of(" \n\r\t");
            return s.substr(start, end - start + 1);
        };

        if (patch.contains("enabled") && patch["enabled"].is_boolean())
        {
            searchConfig_.enabled = patch["enabled"].get<bool>();
        }
        if (patch.contains("endpoints") && patch["endpoints"].is_array())
        {
            searchConfig_.endpoints.clear();
            std::unordered_set<std::string> seen;
            for (auto &v : patch["endpoints"])
            {
                std::string s;
                if (v.is_string())
                    s = v.get<std::string>();
                else if (v.is_number() || v.is_boolean())
                    s = v.dump();
                else if (v.is_null())
                    s.clear();
                else
                    s = v.dump();
                s = trim(s);
                if (!s.empty() && !seen.count(s))
                {
                    seen.insert(s);
                    searchConfig_.endpoints.push_back(s);
                }
            }
        }
        if (patch.contains("active") && patch["active"].is_string())
        {
            searchConfig_.active = trim(patch["active"].get<std::string>());
            if (!searchConfig_.active.empty() && std::find(searchConfig_.endpoints.begin(), searchConfig_.endpoints.end(), searchConfig_.active) == searchConfig_.endpoints.end())
            {
                searchConfig_.endpoints.push_back(searchConfig_.active);
            }
        }

        config_.searchEndpoint = searchConfig_.active;
        if (!researcher_)
            researcher_ = std::make_shared<OnlineResearcher>(this, searchConfig_.active);
        researcher_->setEnabled(searchConfig_.enabled);
        researcher_->setEndpoint(searchConfig_.active);
        return searchConfig_.toJson();
    }

    std::vector<std::string> listRobotsFiles() const override
    {
        return ensureRobotsCorpus()->listFiles();
    }

    std::vector<RobotsDocument> collectRobotsDocuments(const json &options) const override
    {
        return ensureRobotsCorpus()->collect(options);
    }

    std::vector<std::string> lemmatizeTokens(const std::vector<std::string> &tokens)
    {
        return ensureRobotsCorpus()->normalizeWords(tokens);
    }

    json ingestDocument(const json &doc) override
    {
        std::string text = doc.value("text", "");
        std::vector<std::string> tokens;
        if (doc.contains("tokens") && doc["tokens"].is_array())
        {
            std::function<std::string(const json &)> jsStringForJoin;
            jsStringForJoin = [&](const json &v) -> std::string
            {
                if (v.is_null())
                    return "";
                if (v.is_string())
                    return v.get<std::string>();
                if (v.is_boolean())
                    return v.get<bool>() ? "true" : "false";
                if (v.is_number())
                    return v.dump();
                if (v.is_array())
                {
                    std::string out;
                    bool first = true;
                    for (auto &item : v)
                    {
                        if (!first)
                            out += ",";
                        first = false;
                        out += jsStringForJoin(item);
                    }
                    return out;
                }
                return "[object Object]";
            };
            auto jsStringOrEmpty = [&](const json &v) -> std::string
            {
                if (v.is_null())
                    return "";
                if (v.is_boolean())
                    return v.get<bool>() ? "true" : "";
                if (v.is_number())
                {
                    double n = v.get<double>();
                    if (n == 0.0)
                        return "";
                    return v.dump();
                }
                if (v.is_string())
                    return v.get<std::string>();
                if (v.is_array())
                {
                    std::string out;
                    bool first = true;
                    for (auto &item : v)
                    {
                        if (!first)
                            out += ",";
                        first = false;
                        out += jsStringForJoin(item);
                    }
                    return out;
                }
                return "[object Object]";
            };
            for (auto &t : doc["tokens"])
            {
                std::string s = jsStringOrEmpty(t);
                if (!s.empty())
                    tokens.push_back(s);
            }
        }
        if (tokens.empty() && !text.empty())
            tokens = tokenize(text);
        if (text.empty() && tokens.empty())
            return json{{"ok", false}, {"reason", "empty-text"}};
        if (tokens.empty())
            return json{{"ok", false}, {"reason", "no-tokens"}};

        auto memeIds = buildMemeSequenceFromTokens(tokens);
        for (size_t i = 0; i + 1 < memeIds.size(); i++)
        {
            graph_.link(memeIds[i], memeIds[i + 1], 1.0, 0);
        }
        std::string source = doc.value("source", "");
        for (auto &memeId : memeIds)
        {
            json meta = graph_.getMeta(memeId);
            meta["lastTouched"] = nowMs();
            meta["degree"] = graph_.degreeOf(memeId);
            if (!source.empty())
            {
                if (!meta.contains("sources") || !meta["sources"].is_object())
                    meta["sources"] = json::object();
                meta["sources"][source] = meta["sources"].value(source, 0) + 1;
            }
            graph_.setMeta(memeId, meta);
        }
        corpusStats_["ingested"] = corpusStats_.value("ingested", 0) + 1;
        corpusStats_["lastIngest"] = nowMs();
        return json{{"ok", true}, {"memes", (int)memeIds.size()}, {"edges", (int)std::max<int>(0, (int)memeIds.size() - 1)}, {"source", source.empty() ? json(nullptr) : json(source)}};
    }

    json forgetMemes(const json &payload) override
    {
        int64_t threshold = payload.contains("olderThan") ? payload.value("olderThan", 0LL)
                                                          : (nowMs() - 7LL * 24LL * 60LL * 60LL * 1000LL);
        int limit = payload.value("limit", 64);
        json removed = json::array();
        for (auto &kv : graph_.metaEntries())
        {
            auto &meta = kv.second;
            if (meta.value("lastTouched", 0LL) >= threshold)
                continue;
            auto words = kvm_.getMemeWords(kv.first);
            for (auto &w : words)
                kvm_.unlink(w, kv.first);
            if (graph_.removeNode(kv.first))
            {
                json ws = json::array();
                for (auto &w : words)
                    ws.push_back(w);
                removed.push_back(json{{"memeId", kv.first}, {"words", ws}});
            }
            if (limit > 0 && (int)removed.size() >= limit)
                break;
        }
        return json{{"ok", true}, {"removed", removed}};
    }

    json onlineLookup(const std::string &query, const json &options) override
    {
        return onlineLookup(json(query), options);
    }

    json onlineLookup(const json &input, const json &options) override
    {
        if (!researcher_)
        {
            researcher_ = std::make_shared<OnlineResearcher>(this, searchConfig_.active);
            researcher_->setEnabled(searchConfig_.enabled);
        }
        return researcher_->lookup(input, options);
    }

    fs::path exportGraphToFile(const json &options) override
    {
        ensureDir(config_.exportDir);
        auto jsToNumber = [&](const json &v) -> double
        {
            if (v.is_null())
                return std::numeric_limits<double>::quiet_NaN();
            if (v.is_number())
                return v.get<double>();
            if (v.is_boolean())
                return v.get<bool>() ? 1.0 : 0.0;
            if (v.is_string())
            {
                std::string s = v.get<std::string>();
                auto start = s.find_first_not_of(" \n\r\t");
                if (start == std::string::npos)
                    return 0.0;
                auto end = s.find_last_not_of(" \n\r\t");
                s = s.substr(start, end - start + 1);
                if (s.empty())
                    return 0.0;
                try
                {
                    return std::stod(s);
                }
                catch (...)
                {
                    return std::numeric_limits<double>::quiet_NaN();
                }
            }
            return std::numeric_limits<double>::quiet_NaN();
        };
        auto asBool = [](const json &v, bool fallback)
        {
            if (v.is_boolean())
                return v.get<bool>();
            if (v.is_number())
                return v.get<double>() != 0.0;
            if (v.is_string())
            {
                std::string s = v.get<std::string>();
                std::transform(s.begin(), s.end(), s.begin(), [](unsigned char ch)
                               { return (char)std::tolower(ch); });
                if (s == "1" || s == "true" || s == "yes" || s == "on")
                    return true;
                if (s == "0" || s == "false" || s == "no" || s == "off")
                    return false;
            }
            return fallback;
        };

        int radius = 2;
        if (options.contains("radius"))
        {
            double r = jsToNumber(options["radius"]);
            if (!std::isfinite(r))
                radius = 0;
            else
                radius = (int)r;
        }
        radius = std::max(0, std::min(8, radius));

        int topKNodes = 0;
        if (options.contains("topKNodes"))
        {
            double v = jsToNumber(options["topKNodes"]);
            if (std::isfinite(v) && v > 0)
                topKNodes = std::max(1, std::min(100000, (int)v));
        }
        double minActivation = -std::numeric_limits<double>::infinity();
        if (options.contains("minActivation"))
        {
            double v = jsToNumber(options["minActivation"]);
            if (std::isfinite(v))
                minActivation = v;
        }
        std::string format = options.value("format", "json");
        std::transform(format.begin(), format.end(), format.begin(), [](unsigned char ch)
                       { return (char)std::tolower(ch); });
        if (format != "json" && format != "jsonl" && format != "csv")
            format = "json";
        bool includeMeta = asBool(options.value("includeMeta", json()), true);
        bool allowAbsolute = asBool(options.value("allowAbsolutePath", json()), false);

        std::unordered_map<std::string, double> usedSeeds;
        std::vector<std::string> seedOrder;
        if (options.contains("seeds") && options["seeds"].is_array())
        {
            std::vector<std::string> seedWords;
            for (auto &s : options["seeds"])
            {
                if (s.is_string())
                    seedWords.push_back(s.get<std::string>());
                else if (s.is_number())
                    seedWords.push_back(s.dump());
                else if (s.is_boolean())
                    seedWords.push_back(s.get<bool>() ? "true" : "false");
            }
            auto mapping = mapWordsToMemesOrdered(seedWords);
            usedSeeds = mapping.seeds;
            seedOrder = mapping.order;
        }
        else
        {
            auto mapping = mapWordsToMemesOrdered(std::vector<std::string>{});
            usedSeeds = mapping.seeds;
            seedOrder = mapping.order;
        }
        std::vector<std::string> seedIds;
        if (!seedOrder.empty())
        {
            seedIds = seedOrder;
        }
        else
        {
            for (auto &kv : usedSeeds)
                seedIds.push_back(kv.first);
        }
        auto win = graph_.buildWindow(seedIds, radius);
        std::vector<float> seedVec(win.ids.size(), 0.0f);
        for (auto &kv : usedSeeds)
        {
            auto it = win.index.find(kv.first);
            if (it != win.index.end())
                seedVec[it->second] = (float)kv.second;
        }
        auto actFn = activationFn(params_);
        int iteration = params_.iteration ? params_.iteration : 5;
        if (iteration == 0)
            iteration = 5;
        auto activation = tensor_.iteratePropagation(win.csr, seedVec, iteration, actFn, params_.decayK, 0.02);
        auto graphObj = GraphExportBuilder::fromWindow(win, &activation);

        if (graphObj.contains("Nodes") && graphObj["Nodes"].is_array())
        {
            std::vector<json> nodes;
            nodes.reserve(graphObj["Nodes"].size());
            for (auto &n : graphObj["Nodes"])
            {
                if (!n.is_object())
                    continue;
                double value = n.value("value", 0.0);
                if (value < minActivation)
                    continue;
                nodes.push_back(n);
            }
            std::sort(nodes.begin(), nodes.end(), [](const json &a, const json &b)
                      { return a.value("value", 0.0) > b.value("value", 0.0); });
            if (topKNodes > 0 && (int)nodes.size() > topKNodes)
                nodes.resize(topKNodes);

            std::unordered_set<std::string> keepIds;
            for (auto &n : nodes)
            {
                std::string id = n.value("id", "");
                if (!id.empty())
                    keepIds.insert(id);
            }

            json outNodes = json::array();
            for (auto &n : nodes)
                outNodes.push_back(n);
            graphObj["Nodes"] = outNodes;

            if (graphObj.contains("Edges") && graphObj["Edges"].is_array())
            {
                json outEdges = json::array();
                for (auto &e : graphObj["Edges"])
                {
                    if (!e.is_object())
                        continue;
                    std::string from = e.value("from", "");
                    std::string to = e.value("to", "");
                    if (keepIds.count(from) && keepIds.count(to))
                        outEdges.push_back(e);
                }
                graphObj["Edges"] = outEdges;
            }
        }

        if (includeMeta)
        {
            json meta = json::object();
            meta["createdAt"] = nowIso();
            meta["radius"] = radius;
            meta["iteration"] = iteration;
            meta["seedCount"] = (int)usedSeeds.size();
            meta["seeds"] = json::array();
            for (auto &sid : seedIds)
            {
                meta["seeds"].push_back(json{{"id", sid}, {"value", usedSeeds.count(sid) ? usedSeeds[sid] : 0.0}});
            }
            graphObj["Meta"] = meta;
        }

        std::string fileArg = options.value("file", "");
        fs::path outFile;
        if (!fileArg.empty())
        {
            fs::path candidate = fs::path(fileArg);
            if (candidate.is_absolute())
            {
                if (allowAbsolute)
                    outFile = candidate;
                else
                    outFile = config_.exportDir / candidate.filename();
            }
            else
            {
                outFile = config_.exportDir / candidate;
            }
        }
        else
        {
            std::string ts = nowIso();
            std::replace(ts.begin(), ts.end(), ':', '-');
            std::replace(ts.begin(), ts.end(), '.', '-');
            std::string ext = (format == "json" ? ".json" : (format == "jsonl" ? ".jsonl" : ".csv"));
            outFile = config_.exportDir / ("graph_export_" + ts + ext);
        }

        if (outFile.extension().empty())
        {
            std::string ext = (format == "json" ? ".json" : (format == "jsonl" ? ".jsonl" : ".csv"));
            outFile += ext;
        }

        ensureDir(outFile.parent_path());
        fs::path tmp = outFile;
        tmp += ".tmp";

        if (format == "json")
        {
            std::ofstream o(tmp, std::ios::trunc);
            o << graphObj.dump(2);
        }
        else if (format == "jsonl")
        {
            std::ofstream o(tmp, std::ios::trunc);
            if (graphObj.contains("Nodes") && graphObj["Nodes"].is_array())
            {
                for (auto &n : graphObj["Nodes"])
                {
                    o << json{{"type", "node"}, {"data", n}}.dump() << "\n";
                }
            }
            if (graphObj.contains("Edges") && graphObj["Edges"].is_array())
            {
                for (auto &e : graphObj["Edges"])
                {
                    o << json{{"type", "edge"}, {"data", e}}.dump() << "\n";
                }
            }
            if (graphObj.contains("Meta"))
                o << json{{"type", "meta"}, {"data", graphObj["Meta"]}}.dump() << "\n";
        }
        else
        {
            std::ofstream nodesFile(tmp, std::ios::trunc);
            nodesFile << "id,x,y,z,value\n";
            if (graphObj.contains("Nodes") && graphObj["Nodes"].is_array())
            {
                for (auto &n : graphObj["Nodes"])
                {
                    if (!n.is_object())
                        continue;
                    auto esc = [](std::string s)
                    {
                        std::replace(s.begin(), s.end(), '"', '\'');
                        return s;
                    };
                    nodesFile << '"' << esc(n.value("id", "")) << "\","
                              << n.value("x", 0.0) << ","
                              << n.value("y", 0.0) << ","
                              << n.value("z", 0.0) << ","
                              << n.value("value", 0.0) << "\n";
                }
            }
            fs::path edgeFile = outFile;
            edgeFile.replace_extension(".edges.csv");
            std::ofstream edges(edgeFile, std::ios::trunc);
            edges << "from,to,weight,dir\n";
            if (graphObj.contains("Edges") && graphObj["Edges"].is_array())
            {
                for (auto &e : graphObj["Edges"])
                {
                    if (!e.is_object())
                        continue;
                    auto esc = [](std::string s)
                    {
                        std::replace(s.begin(), s.end(), '"', '\'');
                        return s;
                    };
                    edges << '"' << esc(e.value("from", "")) << "\",\""
                          << esc(e.value("to", "")) << "\","
                          << e.value("weight", 0.0) << ","
                          << e.value("dir", 0) << "\n";
                }
            }
        }

        if (fs::exists(outFile))
            fs::remove(outFile);
        fs::rename(tmp, outFile);
        return outFile;
    }

    json learnFromDialog(const json &payload, const json &result) override
    {
        try
        {
            json actualPayload = payload;
            json actualResult = result;
            if (payload.is_object() && payload.contains("payload") && payload.contains("result"))
            {
                if (result.is_null() || (result.is_object() && result.empty()))
                {
                    actualPayload = payload.value("payload", json::object());
                    actualResult = payload.value("result", json::object());
                }
            }
            auto trim = [](const std::string &s)
            {
                auto start = s.find_first_not_of(" \n\r\t");
                if (start == std::string::npos)
                    return std::string();
                auto end = s.find_last_not_of(" \n\r\t");
                return s.substr(start, end - start + 1);
            };
            std::function<std::string(const json &)> jsStringForJoin;
            jsStringForJoin = [&](const json &v) -> std::string
            {
                if (v.is_null())
                    return "";
                if (v.is_string())
                    return v.get<std::string>();
                if (v.is_boolean())
                    return v.get<bool>() ? "true" : "false";
                if (v.is_number())
                    return v.dump();
                if (v.is_array())
                {
                    std::string out;
                    bool first = true;
                    for (auto &item : v)
                    {
                        if (!first)
                            out += ",";
                        first = false;
                        out += jsStringForJoin(item);
                    }
                    return out;
                }
                return "[object Object]";
            };

            std::string question;
            if (actualPayload.contains("text") && actualPayload["text"].is_string())
            {
                question = actualPayload["text"].get<std::string>();
            }
            else if (actualPayload.contains("tokens") && actualPayload["tokens"].is_array())
            {
                std::ostringstream oss;
                bool first = true;
                for (auto &t : actualPayload["tokens"])
                {
                    std::string token = jsStringForJoin(t);
                    if (!first)
                        oss << " ";
                    oss << token;
                    first = false;
                }
                question = oss.str();
            }

            std::string reply = actualResult.contains("reply") && actualResult["reply"].is_string()
                                    ? actualResult["reply"].get<std::string>()
                                    : std::string();

            if (trim(question).empty() || trim(reply).empty())
            {
                return json{{"ok", false}, {"reason", "missing-text"}};
            }
            if (isLikelyGibberishReply(reply))
            {
                return json{{"ok", false}, {"reason", "gibberish-reply-skipped"}};
            }

            double residual = 0.0;
            double residualGate = 1.0;
            try
            {
                auto qEmb = textToMiniEmbedding(question, 128);
                auto rEmb = textToMiniEmbedding(reply, 128);
                double cos = cosineSim(qEmb, rEmb);
                double sim = clamp01((cos + 1.0) * 0.5);
                residual = 1.0 - sim;
                residualGate = std::max(0.15, 1.0 - residual);
                residualStats_.count += 1;
                residualStats_.last = residual;
                if (residualStats_.count == 1)
                {
                    residualStats_.ema = residual;
                }
                else
                {
                    residualStats_.ema = residualStats_.ema * (1.0 - residualStats_.alpha) + residual * residualStats_.alpha;
                }
                if (residualStats_.count % std::max(1, residualStats_.adjustEvery) == 0)
                {
                    if (residualStats_.ema > 0.55)
                    {
                        params_.iteration = std::min(12, std::max(3, params_.iteration) + 1);
                        params_.decayK = std::max(0.5, params_.decayK - 0.05);
                    }
                    else if (residualStats_.ema < 0.30)
                    {
                        params_.iteration = std::max(3, std::max(3, params_.iteration) - 1);
                        params_.decayK = std::min(2.0, params_.decayK + 0.05);
                    }
                }
            }
            catch (...)
            {
                residual = 0.0;
                residualGate = 1.0;
            }

            std::unordered_map<std::string, double> seeds;
            std::vector<std::string> seedOrder;
            auto jsToNumber = [&](const json &v) -> double
            {
                if (v.is_null())
                    return 0.0;
                if (v.is_number())
                    return v.get<double>();
                if (v.is_boolean())
                    return v.get<bool>() ? 1.0 : 0.0;
                if (v.is_string())
                {
                    std::string s = v.get<std::string>();
                    auto start = s.find_first_not_of(" \n\r\t");
                    if (start == std::string::npos)
                        return 0.0;
                    auto end = s.find_last_not_of(" \n\r\t");
                    s = s.substr(start, end - start + 1);
                    if (s.empty())
                        return 0.0;
                    try
                    {
                        return std::stod(s);
                    }
                    catch (...)
                    {
                        return std::numeric_limits<double>::quiet_NaN();
                    }
                }
                return std::numeric_limits<double>::quiet_NaN();
            };

            if (actualResult.contains("seeds") && actualResult["seeds"].is_array())
            {
                for (auto &m : actualResult["seeds"])
                {
                    if (m.is_array() && m.size() >= 2 && m[0].is_string())
                    {
                        std::string id = m[0].get<std::string>();
                        seeds[id] = jsToNumber(m[1]);
                        seedOrder.push_back(id);
                    }
                }
            }
            else
            {
                auto mapping = mapWordsToMemesOrdered(tokenize(question));
                seeds = mapping.seeds;
                seedOrder = mapping.order;
            }

            PropagationResult resObj;
            if (actualResult.contains("memes") && actualResult["memes"].is_array() && actualResult.contains("activation") && actualResult["activation"].is_array())
            {
                for (auto &m : actualResult["memes"])
                    if (m.is_string())
                        resObj.windowInfo.ids.push_back(m.get<std::string>());
                for (size_t i = 0; i < resObj.windowInfo.ids.size(); i++)
                    resObj.windowInfo.index[resObj.windowInfo.ids[i]] = (uint32_t)i;
                for (auto &a : actualResult["activation"])
                    resObj.activation.push_back((float)jsToNumber(a));
            }
            else
            {
                resObj = runPropagation(seeds, seedOrder.empty() ? json::object() : json{{"seedOrder", seedOrder}});
            }

            auto topMemes = pickTopActivatedMemes(resObj, seeds, 18, 1e-6);
            auto signature = makeSignatureFromTopMemes(topMemes, 12);
            std::vector<std::string> memeIds;
            for (auto &m : topMemes)
                memeIds.push_back(m.memeId);

            for (size_t i = 0; i < topMemes.size() && i < 10; i++)
            {
                double w = std::max(0.5, std::min(3.0, topMemes[i].score));
                w = std::max(0.2, w * residualGate);
                surfaceLexicon_->learn(topMemes[i].memeId, reply, w);
            }
            std::string sessionId;
            if (actualPayload.contains("sessionId") && actualPayload["sessionId"].is_string())
                sessionId = actualPayload["sessionId"].get<std::string>();
            if (sessionId.empty())
                return json{{"ok", false}, {"reason", "missing-session"}};

            double hint = topMemes.empty() ? 0.0 : topMemes.front().score;
            dialogMemory_->remember(sessionId, signature, memeIds, question, reply, hint * residualGate);
            int sigLen = 0;
            if (!signature.empty())
            {
                sigLen = 1;
                for (char c : signature)
                    if (c == '|')
                        sigLen++;
            }
            return json{{"ok", true}, {"memes", (int)memeIds.size()}, {"signatureLen", sigLen}, {"residual", residual}, {"residualEma", residualStats_.ema}, {"residualGate", residualGate}};
        }
        catch (const std::exception &e)
        {
            return json{{"ok", false}, {"error", e.what()}};
        }
    }

private:
    friend class OnlineResearcher;
    Config config_;
    KVMStore kvm_;
    MemeGraph graph_;
    TensorEngine tensor_;
    std::shared_ptr<KeyValueStore> sessions_;
    std::shared_ptr<SessionManager> sessionManager_;
    std::shared_ptr<NamespacedStore> surfaceStore_;
    std::shared_ptr<NamespacedStore> dialogStore_;
    std::shared_ptr<MemeSurfaceLexicon> surfaceLexicon_;
    std::shared_ptr<DialogMemory> dialogMemory_;
    mutable std::shared_ptr<RobotsCorpus> robotsCorpus_;
    ModelDefaults params_;
    struct ResidualStats
    {
        double ema{0.0};
        double last{0.0};
        double alpha{0.05};
        int64_t count{0};
        int adjustEvery{20};
    } residualStats_;
    json metrics_ = json{{"requests", 0}, {"lastLatency", 0}, {"updatedAt", 0}};
    json corpusStats_ = json{{"ingested", 0}, {"lastIngest", nullptr}};
    SearchConfig searchConfig_;
    std::shared_ptr<OnlineResearcher> researcher_;
    PatternMatrix pattern_;
    std::shared_ptr<addon::AddonManager> addons_;

    struct PropagationResult
    {
        MemeGraph::WindowInfo windowInfo;
        std::vector<float> seedVector;
        std::vector<float> activation;
    };

    struct TopMeme
    {
        std::string memeId;
        double score{0.0};
    };

    std::shared_ptr<RobotsCorpus> ensureRobotsCorpus() const
    {
        if (!robotsCorpus_)
        {
            robotsCorpus_ = std::make_shared<RobotsCorpus>(config_.robotsDir, config_.lemmaCsv, config_.lemmaAutoload,
                                                           config_.lemmaMaxBytes, config_.lemmaForce,
                                                           config_.robotsChunkMinWords, config_.robotsChunkMaxWords);
        }
        return robotsCorpus_;
    }

    static int64_t nowMs()
    {
        return (int64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::system_clock::now().time_since_epoch())
            .count();
    }

    PropagationResult runPropagation(const std::unordered_map<std::string, double> &seeds, const json &options)
    {
        PropagationResult result;
        auto jsToNumber = [&](const json &v) -> double
        {
            if (v.is_null())
                return std::numeric_limits<double>::quiet_NaN();
            if (v.is_number())
                return v.get<double>();
            if (v.is_boolean())
                return v.get<bool>() ? 1.0 : 0.0;
            if (v.is_string())
            {
                std::string s = v.get<std::string>();
                auto start = s.find_first_not_of(" \n\r\t");
                if (start == std::string::npos)
                    return 0.0;
                auto end = s.find_last_not_of(" \n\r\t");
                s = s.substr(start, end - start + 1);
                if (s.empty())
                    return 0.0;
                try
                {
                    return std::stod(s);
                }
                catch (...)
                {
                    return std::numeric_limits<double>::quiet_NaN();
                }
            }
            return std::numeric_limits<double>::quiet_NaN();
        };
        auto numberOrHard = [&](const json *v, double paramFallback, double hardFallback) -> double
        {
            double n = paramFallback;
            if (v)
                n = jsToNumber(*v);
            if (!std::isfinite(n) || n == 0.0)
                return hardFallback;
            return n;
        };
        const json *radiusVal = options.contains("radius") && !options["radius"].is_null()
                                    ? &options["radius"]
                                    : (options.contains("windowRadius") && !options["windowRadius"].is_null() ? &options["windowRadius"] : nullptr);
        int radius = std::max(1, std::min(6, (int)numberOrHard(radiusVal, 2, 2)));
        int iteration = std::max(1, (int)numberOrHard(options.contains("iteration") && !options["iteration"].is_null() ? &options["iteration"] : nullptr,
                                                      params_.iteration ? params_.iteration : 5, 5));
        std::vector<std::string> seedIds;
        if (options.contains("seedOrder") && options["seedOrder"].is_array())
        {
            for (auto &v : options["seedOrder"])
                if (v.is_string())
                    seedIds.push_back(v.get<std::string>());
        }
        if (seedIds.empty())
        {
            seedIds.reserve(seeds.size());
            for (auto &kv : seeds)
                seedIds.push_back(kv.first);
        }
        result.windowInfo = graph_.buildWindow(seedIds, radius);
        result.seedVector.assign(result.windowInfo.ids.size(), 0.0f);
        for (auto &kv : seeds)
        {
            auto it = result.windowInfo.index.find(kv.first);
            if (it != result.windowInfo.index.end())
                result.seedVector[it->second] = (float)kv.second;
        }
        auto actFn = activationFn(params_);
        result.activation = tensor_.iteratePropagation(result.windowInfo.csr, result.seedVector, iteration, actFn, params_.decayK, 0.02);
        pattern_.rebuild(result.windowInfo);
        return result;
    }

    std::vector<TopMeme> pickTopActivatedMemes(const PropagationResult &result,
                                               const std::unordered_map<std::string, double> &seeds,
                                               int limit,
                                               double minScore)
    {
        std::vector<TopMeme> scored;
        std::unordered_set<std::string> seedIds;
        for (auto &kv : seeds)
            seedIds.insert(kv.first);

        auto isConnectedToSeeds = [&](const std::string &memeId)
        {
            if (seedIds.empty() || seedIds.count(memeId))
                return true;
            for (auto &n : graph_.neighborsOf(memeId))
            {
                if (seedIds.count(n.first))
                    return true;
            }
            for (auto &sid : seedIds)
            {
                for (auto &n : graph_.neighborsOf(sid))
                {
                    if (n.first == memeId)
                        return true;
                }
            }
            return false;
        };

        for (size_t i = 0; i < result.windowInfo.ids.size(); i++)
        {
            if (i >= result.activation.size())
                break;
            double score = result.activation[i];
            if (!std::isfinite(score) || score <= minScore)
                continue;
            const auto &memeId = result.windowInfo.ids[i];
            if (!isConnectedToSeeds(memeId))
                continue;
            scored.push_back({memeId, score});
        }
        std::sort(scored.begin(), scored.end(), [](auto &a, auto &b)
                  { return a.score > b.score; });
        if ((int)scored.size() > limit)
            scored.resize(limit);
        return scored;
    }

    std::string makeSignatureFromTopMemes(const std::vector<TopMeme> &topMemes, int limit)
    {
        std::vector<std::string> ids;
        int cap = std::max(3, limit);
        for (auto &m : topMemes)
        {
            if (!m.memeId.empty())
                ids.push_back(m.memeId);
            if ((int)ids.size() >= cap)
                break;
        }
        std::sort(ids.begin(), ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
        std::ostringstream oss;
        for (size_t i = 0; i < ids.size(); i++)
        {
            if (i)
                oss << "|";
            oss << ids[i];
        }
        return oss.str();
    }

    std::vector<std::string> buildMemeSequenceFromTokens(const std::vector<std::string> &tokens)
    {
        std::vector<std::string> list;
        auto trim = [](const std::string &s)
        {
            auto start = s.find_first_not_of(" \n\r\t");
            if (start == std::string::npos)
                return std::string();
            auto end = s.find_last_not_of(" \n\r\t");
            return s.substr(start, end - start + 1);
        };
        for (auto &t : tokens)
        {
            std::string s = trim(t);
            if (!s.empty())
                list.push_back(s);
        }
        int nMin = std::max(2, params_.memeNgramMin);
        int nMax = std::max(nMin, params_.memeNgramMax);
        int minOverlap = std::max(1, params_.minOverlapThreshold);
        int maxWordSet = std::max(4, params_.maxMemeWords);

        auto resolveOrCreate = [&](const std::vector<std::string> &tokenSet) -> std::string
        {
            std::vector<std::string> uniq;
            std::unordered_set<std::string> seen;
            for (auto &t : tokenSet)
            {
                if (t.empty())
                    continue;
                if (seen.insert(t).second)
                    uniq.push_back(t);
            }
            if ((int)uniq.size() > maxWordSet)
                uniq.resize(maxWordSet);
            if (uniq.size() <= 1)
                return uniq.empty() ? "" : ("meme_" + sha1Hex(uniq[0]));

            std::unordered_map<std::string, int> counts;
            for (auto &w : uniq)
            {
                for (auto &mid : kvm_.getWordMemeSet(w))
                    counts[mid]++;
            }
            std::string best;
            int bestOverlap = 0;
            for (auto &kv : counts)
            {
                if (kv.second > bestOverlap)
                {
                    bestOverlap = kv.second;
                    best = kv.first;
                }
            }
            if (!best.empty() && bestOverlap >= minOverlap)
            {
                for (auto &w : uniq)
                    kvm_.link(w, best);
                graph_.ensureNode(best);
                return best;
            }
            std::vector<std::string> sorted = uniq;
            std::sort(sorted.begin(), sorted.end());
            std::string memeId = "meme_p_" + sha1Hex([&]()
                                                     {
                std::string join; for (size_t i = 0; i < sorted.size(); i++) { if (i) join += "|"; join += sorted[i]; } return join; }());
            graph_.ensureNode(memeId);
            for (auto &w : sorted)
                kvm_.link(w, memeId);
            return memeId;
        };

        std::vector<std::string> seq;
        for (size_t i = 0; i < list.size(); i++)
        {
            std::string picked;
            for (int n = nMax; n >= nMin; n--)
            {
                if (i + n > list.size())
                    continue;
                picked = resolveOrCreate(std::vector<std::string>(list.begin() + i, list.begin() + i + n));
                if (!picked.empty())
                    break;
            }
            if (picked.empty())
            {
                auto &w = list[i];
                if (!w.empty())
                {
                    picked = "meme_" + sha1Hex(w);
                    graph_.ensureNode(picked);
                    kvm_.link(w, picked);
                }
            }
            if (!picked.empty() && (seq.empty() || seq.back() != picked))
                seq.push_back(picked);
        }
        return seq;
    }

    std::string composeReply(const std::string &sessionId,
                             const PropagationResult &result,
                             const std::vector<std::string> &words,
                             const std::unordered_map<std::string, double> &seeds)
    {
        auto topMemes = pickTopActivatedMemes(result, seeds, 18, 1e-6);
        std::string signature = makeSignatureFromTopMemes(topMemes, 12);

        std::vector<std::string> topIds;
        for (auto &m : topMemes)
            topIds.push_back(m.memeId);
        json memoryHit = dialogMemory_ ? dialogMemory_->retrieve(sessionId, topIds, signature) : json();
        if (!memoryHit.is_null() && memoryHit.contains("reply") && memoryHit["reply"].is_string())
        {
            auto r = memoryHit["reply"].get<std::string>();
            auto start = r.find_first_not_of(" \n\r\t");
            if (start != std::string::npos)
            {
                auto end = r.find_last_not_of(" \n\r\t");
                r = r.substr(start, end - start + 1);
            }
            else
            {
                r.clear();
            }
            if (!r.empty())
                return r;
        }

        std::unordered_map<std::string, double> phraseScores;
        for (auto &m : topMemes)
        {
            for (auto &p : surfaceLexicon_->getTop(m.memeId, 4))
            {
                std::string key = p.phrase;
                auto start = key.find_first_not_of(" \n\r\t");
                if (start == std::string::npos)
                    continue;
                auto end = key.find_last_not_of(" \n\r\t");
                key = key.substr(start, end - start + 1);
                if (key.empty())
                    continue;
                phraseScores[key] += std::max(0.0, m.score) * (0.5 + std::max(0.0, p.weight));
            }
        }
        if (!phraseScores.empty())
        {
            std::vector<std::pair<std::string, double>> ordered(phraseScores.begin(), phraseScores.end());
            std::sort(ordered.begin(), ordered.end(), [](auto &a, auto &b)
                      { return a.second > b.second; });
            if (ordered.size() > 2)
                ordered.resize(2);
            std::string out;
            for (size_t i = 0; i < ordered.size(); i++)
            {
                if (i)
                    out += "。";
                out += ordered[i].first;
            }
            return out;
        }

        std::unordered_set<std::string> seedIds;
        for (auto &kv : seeds)
            seedIds.insert(kv.first);
        auto isConnectedToSeeds = [&](const std::string &memeId)
        {
            if (seedIds.empty() || seedIds.count(memeId))
                return true;
            for (auto &n : graph_.neighborsOf(memeId))
            {
                if (seedIds.count(n.first))
                    return true;
            }
            for (auto &sid : seedIds)
            {
                for (auto &n : graph_.neighborsOf(sid))
                {
                    if (n.first == memeId)
                        return true;
                }
            }
            return false;
        };

        std::unordered_map<std::string, double> candidateScores;
        for (size_t i = 0; i < result.windowInfo.ids.size(); i++)
        {
            if (i >= result.activation.size())
                break;
            double score = result.activation[i];
            if (!std::isfinite(score) || score <= 0)
                continue;
            const auto &memeId = result.windowInfo.ids[i];
            if (!isConnectedToSeeds(memeId))
                continue;
            for (auto &w : kvm_.getMemeWords(memeId))
            {
                if (!w.empty())
                    candidateScores[w] = std::max(candidateScores[w], score);
            }
        }
        std::vector<std::string> ordered;
        for (auto &kv : candidateScores)
            ordered.push_back(kv.first);
        std::sort(ordered.begin(), ordered.end(), [&](auto &a, auto &b)
                  { return candidateScores[a] > candidateScores[b]; });

        std::vector<std::string> finalWords;
        std::unordered_set<std::string> seen;
        auto pushWord = [&](const std::string &w)
        { if (!w.empty() && !seen.count(w)) { seen.insert(w); finalWords.push_back(w); } };
        for (auto &w : words)
            pushWord(w);
        for (auto &w : ordered)
        {
            pushWord(w);
            if (finalWords.size() >= 30)
                break;
        }
        if (finalWords.empty())
            return "I need more context to respond.";
        std::ostringstream oss;
        for (size_t i = 0; i < finalWords.size() && i < 30; i++)
        {
            if (i)
                oss << " ";
            oss << finalWords[i];
        }
        return oss.str();
    }
};

// ------------------ MemeBarrier ------------------

// MemeBarrier 负责图谱安全巡检与可疑节点隔离。
// 调用方式：启动后后台周期扫描，并输出统计信息。
// 实现思路：结合图统计、文本模型与阈值策略判定。
// 注意事项：阈值过低会带来误杀，需按业务调优。
// 注意事项：依赖运行时状态，空运行时会自动降级。
// 注意事项：该模块不替代外部安全审计体系。
class MemeBarrier
{
public:
    struct Stats
    {
        int scans{0};
        int isolated{0};
        int64_t lastScanTime{0};
        std::vector<json> lastIsolated;
        int evaluated{0};
        int candidates{0};
        double avgScore{0.0};
        double maxScore{0.0};
        double p90Score{0.0};
        double lastThreshold{0.0};
        double avgDegree{0.0};
        double stdDegree{0.0};
        double avgOutDegree{0.0};
        double stdOutDegree{0.0};
        std::vector<int> scoreHistogram;
    };

    MemeBarrier(std::shared_ptr<RuntimeState> runtime,
                int scanIntervalMs = 10000,
                double maliciousThreshold = 0.7,
                int maxIsolatePerScan = 5)
        : runtime_(std::move(runtime)), scanIntervalMs_(scanIntervalMs),
          maliciousThreshold_(maliciousThreshold), baseThreshold_(maliciousThreshold),
          maxIsolatePerScan_(maxIsolatePerScan) {}

    void start()
    {
        if (running_)
            return;
        running_ = true;
        worker_ = std::thread([this]()
                              {
            while (running_) {
                try { scanNetwork(); } catch (...) {}
                std::this_thread::sleep_for(std::chrono::milliseconds(scanIntervalMs_));
            } });
    }

    void stop()
    {
        running_ = false;
        if (worker_.joinable())
            worker_.join();
    }

    bool running() const { return running_; }

    void setThreshold(double t)
    {
        baseThreshold_ = t;
        maliciousThreshold_ = t;
    }
    double threshold() const { return maliciousThreshold_; }

    void setDisableRateTarget(double rate)
    {
        if (!std::isfinite(rate))
            return;
        disableRateTarget_ = std::max(0.001, std::min(0.2, rate));
    }
    double disableRateTarget() const { return disableRateTarget_; }

    void setUseTextCNN(bool on) { useTextCNN_ = on; }
    void setUseRecommender(bool on) { useRecommender_ = on; }
    void setUseTorchModels(bool on) { useTorchModels_ = on; }
    bool useTextCNN() const { return useTextCNN_; }
    bool useRecommender() const { return useRecommender_; }
    bool useTorchModels() const { return useTorchModels_; }

    Stats getStats() const { return stats_; }

private:
    struct NodeStats
    {
        int degree{0};
        int outDegree{0};
        int selfLoops{0};
    };

    std::shared_ptr<RuntimeState> runtime_;
    int scanIntervalMs_{10000};
    double maliciousThreshold_{0.7};
    double baseThreshold_{0.7};
    double minThreshold_{0.35};
    double maxThreshold_{0.95};
    double thresholdMomentum_{0.15};
    double minIsolationMargin_{0.05};
    int minConsecutiveHits_{2};
    bool adaptiveEnabled_{true};
    bool anomalyEnabled_{true};
    int maxIsolatePerScan_{5};
    std::atomic<bool> running_{false};
    std::thread worker_;
    Stats stats_;
    std::deque<float> scoreWindow_;
    size_t scoreWindowSize_{512};
    std::unordered_map<std::string, int> hitCounts_;
    bool useTextCNN_{true};
    bool useRecommender_{true};
    bool useTorchModels_{true};
    double disableRateTarget_{0.01};
    int recommenderTopics_{24};
    int recommenderDim_{16};
    std::once_flag recommenderInit_;
    std::vector<std::vector<float>> topicVectors_;
    std::unordered_map<std::string, int> corpusFreq_;
    int corpusTotal_{0};
    int corpusMaxFreq_{1};

#ifdef HAVE_TORCH
    struct TorchVocab
    {
        int64_t padId{0};
        int64_t unkId{1};
        int64_t maxVocab{5000};
        std::unordered_map<std::string, int64_t> stoi;
        std::vector<std::string> itos;

        void build(const std::unordered_map<std::string, int> &freq)
        {
            std::vector<std::pair<std::string, int>> items(freq.begin(), freq.end());
            std::sort(items.begin(), items.end(), [](auto &a, auto &b)
                      { return a.second > b.second; });
            itos.clear();
            stoi.clear();
            itos.push_back("<pad>");
            itos.push_back("<unk>");
            for (size_t i = 0; i < items.size() && (int64_t)itos.size() < maxVocab; i++)
            {
                stoi[items[i].first] = (int64_t)itos.size();
                itos.push_back(items[i].first);
            }
        }

        std::vector<int64_t> encode(const std::vector<std::string> &tokens, int64_t maxLen) const
        {
            std::vector<int64_t> ids(maxLen, padId);
            int64_t idx = 0;
            for (const auto &t : tokens)
            {
                if (idx >= maxLen)
                    break;
                auto it = stoi.find(t);
                ids[idx++] = (it == stoi.end()) ? unkId : it->second;
            }
            return ids;
        }
    };

    struct TextCNNImpl : torch::nn::Module
    {
        torch::nn::Embedding emb{nullptr};
        torch::nn::Conv1d conv3{nullptr};
        torch::nn::Conv1d conv4{nullptr};
        torch::nn::Conv1d conv5{nullptr};
        torch::nn::Linear fc{nullptr};

        TextCNNImpl(int64_t vocab, int64_t embDim, int64_t outDim)
            : emb(register_module("emb", torch::nn::Embedding(vocab, embDim))),
              conv3(register_module("conv3", torch::nn::Conv1d(torch::nn::Conv1dOptions(embDim, 64, 3).padding(1)))),
              conv4(register_module("conv4", torch::nn::Conv1d(torch::nn::Conv1dOptions(embDim, 64, 4).padding(2)))),
              conv5(register_module("conv5", torch::nn::Conv1d(torch::nn::Conv1dOptions(embDim, 64, 5).padding(2)))),
              fc(register_module("fc", torch::nn::Linear(64 * 3, outDim))) {}

        torch::Tensor forward(torch::Tensor x)
        {
            auto e = emb(x).transpose(1, 2);
            auto c3 = torch::relu(conv3(e));
            auto c4 = torch::relu(conv4(e));
            auto c5 = torch::relu(conv5(e));
            auto p3 = torch::adaptive_max_pool1d(c3, 1).squeeze(-1);
            auto p4 = torch::adaptive_max_pool1d(c4, 1).squeeze(-1);
            auto p5 = torch::adaptive_max_pool1d(c5, 1).squeeze(-1);
            auto cat = torch::cat({p3, p4, p5}, 1);
            auto out = fc(cat);
            return torch::sigmoid(out);
        }
    };
    TORCH_MODULE(TextCNN);

    struct TextRNNImpl : torch::nn::Module
    {
        torch::nn::Embedding emb{nullptr};
        torch::nn::LSTM lstm{nullptr};
        torch::nn::Linear fc{nullptr};

        TextRNNImpl(int64_t vocab, int64_t embDim, int64_t hidDim, int64_t outDim)
            : emb(register_module("emb", torch::nn::Embedding(vocab, embDim))),
              lstm(register_module("lstm", torch::nn::LSTM(torch::nn::LSTMOptions(embDim, hidDim).batch_first(true)))),
              fc(register_module("fc", torch::nn::Linear(hidDim, outDim))) {}

        torch::Tensor forward(torch::Tensor x)
        {
            auto e = emb(x);
            auto out = std::get<0>(lstm(e));
            auto last = out.select(1, out.size(1) - 1);
            auto logits = fc(last);
            return torch::sigmoid(logits);
        }
    };
    TORCH_MODULE(TextRNN);

    struct TorchBundle
    {
        bool ready{false};
        TorchVocab vocab;
        int64_t maxLen{48};
        int64_t embDim{64};
        int64_t hidDim{64};
        TextCNN cnn{nullptr};
        TextRNN rnn{nullptr};
    };

    std::once_flag torchInit_;
    TorchBundle torch_;
#endif

    double evaluateMaliciousness(const std::string &memeId, const NodeStats &ns, double avgConn,
                                 double avgOut, double stdDeg, double stdOut)
    {
        if (!runtime_)
            return 0.0;
        double outSkew = avgConn > 0 ? std::min(1.0, ns.outDegree / (avgConn * 3.0)) : 0;
        double selfSkew = ns.degree > 0 ? std::min(1.0, (double)ns.selfLoops / ns.degree) : 0;
        double zDeg = (stdDeg > 1e-6) ? (ns.degree - avgConn) / stdDeg : 0.0;
        double zOut = (stdOut > 1e-6) ? (ns.outDegree - avgOut) / stdOut : 0.0;
        double anomaly = anomalyEnabled_ ? clamp01((std::max(0.0, zDeg) + std::max(0.0, zOut)) / 6.0) : 0.0;
        double growth = 0; // 暂无 access log，对齐 JS 的可选逻辑
        double score = 0.5 * growth + 0.25 * outSkew + 0.15 * selfSkew + 0.10 * anomaly;
        return clamp01(score);
    }

    static float hashToUnit(const std::string &s)
    {
        uint32_t h = 2166136261u;
        for (unsigned char c : s)
        {
            h ^= c;
            h *= 16777619u;
        }
        return (float)(h % 10000) / 10000.0f;
    }

    std::vector<float> wordVector(const std::string &word, int dim) const
    {
        std::vector<float> v(dim, 0.0f);
        std::seed_seq seed{static_cast<unsigned int>(std::hash<std::string>{}(word))};
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
        for (int i = 0; i < dim; i++)
            v[i] = dist(rng);
        return v;
    }

    std::vector<float> embedTokens(const std::vector<std::string> &tokens, int dim) const
    {
        std::vector<float> out(dim, 0.0f);
        int count = 0;
        for (const auto &t : tokens)
        {
            if (t.empty())
                continue;
            auto v = wordVector(t, dim);
            for (int i = 0; i < dim; i++)
                out[i] += v[i];
            count++;
        }
        if (count > 0)
        {
            for (int i = 0; i < dim; i++)
                out[i] /= (float)count;
        }
        return out;
    }

    void ensureRecommenderReady()
    {
        if (!runtime_ || !useRecommender_)
            return;
        std::call_once(recommenderInit_, [&]()
                       {
            try {
                auto docs = runtime_->collectRobotsDocuments(json{{"limit", 200}, {"shuffle", true}});
                for (const auto &doc : docs) {
                    for (const auto &t : doc.tokens) {
                        if (t.empty()) continue;
                        corpusFreq_[t] += 1;
                        corpusTotal_ += 1;
                        corpusMaxFreq_ = std::max(corpusMaxFreq_, corpusFreq_[t]);
                    }
                }
                for (const auto &doc : docs) {
                    if (doc.tokens.empty()) continue;
                    topicVectors_.push_back(embedTokens(doc.tokens, recommenderDim_));
                    if ((int)topicVectors_.size() >= recommenderTopics_) break;
                }
            } catch (...) {
            } });
    }

#ifdef HAVE_TORCH
    void ensureTorchReady()
    {
        if (!runtime_ || !useTorchModels_)
            return;
        std::call_once(torchInit_, [&]()
                       {
            try {
                ensureRecommenderReady();
                auto docs = runtime_->collectRobotsDocuments(json{{"limit", 400}, {"shuffle", true}});
                torch_.vocab.build(corpusFreq_);
                int64_t vocabSize = std::max<int64_t>(2, (int64_t)torch_.vocab.itos.size());
                torch_.cnn = TextCNN(vocabSize, torch_.embDim, 1);
                torch_.rnn = TextRNN(vocabSize, torch_.embDim, torch_.hidDim, 1);
                torch_.cnn->train();
                torch_.rnn->train();

                std::vector<std::vector<int64_t>> xs;
                std::vector<float> ys;
                xs.reserve(docs.size());
                ys.reserve(docs.size());
                for (const auto &doc : docs) {
                    if (doc.tokens.empty()) continue;
                    xs.push_back(torch_.vocab.encode(doc.tokens, torch_.maxLen));
                    ys.push_back((float)computeCopyFeatures(doc.tokens));
                    if (xs.size() >= 512) break;
                }
                if (xs.empty()) {
                    torch_.ready = true;
                    return;
                }

                auto xTensor = torch::zeros({(int64_t)xs.size(), torch_.maxLen}, torch::kInt64);
                for (size_t i = 0; i < xs.size(); i++) {
                    for (int64_t j = 0; j < torch_.maxLen; j++) {
                        xTensor.index_put_({(int64_t)i, j}, xs[i][(size_t)j]);
                    }
                }
                auto yTensor = torch::from_blob(ys.data(), {(int64_t)ys.size(), 1}, torch::kFloat32).clone();

                torch::optim::Adam optCnn(torch_.cnn->parameters(), torch::optim::AdamOptions(1e-3));
                torch::optim::Adam optRnn(torch_.rnn->parameters(), torch::optim::AdamOptions(1e-3));
                const int epochs = 2;
                const int batch = 32;
                for (int e = 0; e < epochs; e++) {
                    for (int64_t i = 0; i < xTensor.size(0); i += batch) {
                        int64_t end = std::min<int64_t>(xTensor.size(0), i + batch);
                        auto xb = xTensor.slice(0, i, end);
                        auto yb = yTensor.slice(0, i, end);

                        optCnn.zero_grad();
                        auto predC = torch_.cnn->forward(xb);
                        auto lossC = torch::mse_loss(predC, yb);
                        lossC.backward();
                        optCnn.step();

                        optRnn.zero_grad();
                        auto predR = torch_.rnn->forward(xb);
                        auto lossR = torch::mse_loss(predR, yb);
                        lossR.backward();
                        optRnn.step();
                    }
                }
                torch_.cnn->eval();
                torch_.rnn->eval();
                torch_.ready = true;
            } catch (...) {
            } });
    }

    double torchCnnScore(const std::vector<std::string> &tokens)
    {
        if (!useTorchModels_)
            return 0.0;
        ensureTorchReady();
        if (!torch_.ready || !torch_.cnn)
            return 0.0;
        auto ids = torch_.vocab.encode(tokens, torch_.maxLen);
        auto x = torch::zeros({1, torch_.maxLen}, torch::kInt64);
        for (int64_t j = 0; j < torch_.maxLen; j++)
            x.index_put_({0, j}, ids[(size_t)j]);
        auto y = torch_.cnn->forward(x);
        return clamp01(y.item<double>());
    }

    double torchRnnScore(const std::vector<std::string> &tokens)
    {
        if (!useTorchModels_)
            return 0.0;
        ensureTorchReady();
        if (!torch_.ready || !torch_.rnn)
            return 0.0;
        auto ids = torch_.vocab.encode(tokens, torch_.maxLen);
        auto x = torch::zeros({1, torch_.maxLen}, torch::kInt64);
        for (int64_t j = 0; j < torch_.maxLen; j++)
            x.index_put_({0, j}, ids[(size_t)j]);
        auto y = torch_.rnn->forward(x);
        return clamp01(y.item<double>());
    }
#endif

    double cosineSimVec(const std::vector<float> &a, const std::vector<float> &b) const
    {
        if (a.empty() || b.empty() || a.size() != b.size())
            return 0.0;
        double dot = 0.0, na = 0.0, nb = 0.0;
        for (size_t i = 0; i < a.size(); i++)
        {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        if (na <= 0.0 || nb <= 0.0)
            return 0.0;
        return dot / (std::sqrt(na) * std::sqrt(nb));
    }

    double computeRecommenderScore(const std::vector<std::string> &tokens)
    {
        if (!useRecommender_)
            return 0.0;
        ensureRecommenderReady();
        if (topicVectors_.empty())
            return 0.0;
        auto emb = embedTokens(tokens, recommenderDim_);
        double best = 0.0;
        for (const auto &topic : topicVectors_)
        {
            best = std::max(best, cosineSimVec(emb, topic));
        }
        double risk = 1.0 - clamp01(best);
        return clamp01(risk);
    }

    double computeCnnScore(const std::vector<std::string> &tokens) const
    {
        if (!useTextCNN_)
            return 0.0;
        if (tokens.empty())
            return 0.0;

        const int side = 24;
        std::vector<float> grid((size_t)side * side, 0.0f);
        std::vector<float> grid2((size_t)side * side, 0.0f);

        for (size_t i = 0; i < tokens.size(); i++)
        {
            const std::string &token = tokens[i];
            if (token.empty())
                continue;
            uint32_t h1 = hashStrSimple(token, 2166136261u);
            uint32_t h2 = hashStrSimple(token, 16777619u);
            int r = (int)(h1 % side);
            int c = (int)((h1 / (uint32_t)side + h2) % side);
            double posDecay = 1.0 / (1.0 + 0.02 * (double)i);
            double lenGain = 1.0 + 0.1 * std::log1p((double)token.size());
            float v = (float)((0.4 + 0.6 * hashToUnit(token)) * posDecay * lenGain);
            grid[(size_t)r * side + c] += v;

            int r2 = (int)((h2 + i) % (uint32_t)side);
            int c2 = (int)((h2 / (uint32_t)side + h1 + 3 * i) % (uint32_t)side);
            grid2[(size_t)r2 * side + c2] += (float)(0.5 * v);
        }

        auto normalizeGrid = [&](std::vector<float> &g)
        {
            if (g.empty())
                return;
            double mean = 0.0;
            for (float x : g)
                mean += x;
            mean /= (double)g.size();
            double var = 0.0;
            for (float x : g)
            {
                double d = x - mean;
                var += d * d;
            }
            var /= (double)g.size();
            double stdv = std::sqrt(std::max(1e-9, var));
            for (auto &x : g)
                x = (float)((x - mean) / stdv);
        };
        normalizeGrid(grid);
        normalizeGrid(grid2);

        auto convStats = [&](const std::vector<float> &g, const float *k, int stride, double &avg, double &mx, double &energy)
        {
            avg = 0.0;
            mx = -1e9;
            energy = 0.0;
            int cnt = 0;
            for (int r = 0; r < side; r += std::max(1, stride))
            {
                for (int c = 0; c < side; c += std::max(1, stride))
                {
                    double sum = 0.0;
                    for (int dr = -1; dr <= 1; dr++)
                    {
                        for (int dc = -1; dc <= 1; dc++)
                        {
                            int rr = std::max(0, std::min(side - 1, r + dr));
                            int cc = std::max(0, std::min(side - 1, c + dc));
                            sum += g[(size_t)rr * side + cc] * k[(dr + 1) * 3 + (dc + 1)];
                        }
                    }
                    sum = std::max(0.0, sum);
                    avg += sum;
                    mx = std::max(mx, sum);
                    energy += sum * sum;
                    cnt++;
                }
            }
            if (cnt > 0)
            {
                avg /= cnt;
                energy = std::sqrt(energy / cnt);
            }
            else
            {
                avg = 0.0;
                mx = 0.0;
                energy = 0.0;
            }
        };

        const float kEdge[9] = {-1, -1, -1, -1, 8, -1, -1, -1, -1};
        const float kBlob[9] = {1, 2, 1, 2, 4, 2, 1, 2, 1};
        const float kDiag[9] = {2, -1, 0, -1, 2, -1, 0, -1, 2};

        double avg1 = 0, max1 = 0, e1 = 0;
        double avg2 = 0, max2 = 0, e2 = 0;
        double avg3 = 0, max3 = 0, e3 = 0;
        convStats(grid, kEdge, 1, avg1, max1, e1);
        convStats(grid, kBlob, 2, avg2, max2, e2);
        convStats(grid2, kDiag, 1, avg3, max3, e3);

        std::unordered_set<std::string> uniq(tokens.begin(), tokens.end());
        double repeatRisk = 1.0 - (double)uniq.size() / std::max(1.0, (double)tokens.size());
        double convRisk = 0.20 * clamp01(avg1 / 3.0) + 0.18 * clamp01(max1 / 5.0) +
                          0.18 * clamp01(e1 / 4.0) + 0.14 * clamp01(avg2 / 3.0) +
                          0.12 * clamp01(max2 / 4.5) + 0.10 * clamp01(e2 / 3.5) +
                          0.08 * clamp01((avg3 + max3 + e3) / 10.0);

        double mixed = 0.78 * convRisk + 0.22 * repeatRisk;
        return clamp01(mixed);
    }

    double computeCopyFeatures(const std::vector<std::string> &tokens)
    {
        if (tokens.empty())
            return 0.0;
        std::unordered_set<std::string> uniq(tokens.begin(), tokens.end());
        double variability = tokens.empty() ? 0.0 : (double)uniq.size() / std::max<size_t>(1, tokens.size());
        double rep = 0.0;
        double sel = 0.0;
        double normIdf = std::log(1.0 + std::max(1, corpusTotal_));
        for (const auto &t : uniq)
        {
            int f = 0;
            auto it = corpusFreq_.find(t);
            if (it != corpusFreq_.end())
                f = it->second;
            rep += (double)f / std::max(1, corpusMaxFreq_);
            double idf = std::log(1.0 + (double)std::max(1, corpusTotal_) / (1.0 + f));
            sel += (normIdf > 0.0 ? idf / normIdf : 0.0);
        }
        rep = rep / std::max<size_t>(1, uniq.size());
        sel = sel / std::max<size_t>(1, uniq.size());
        double score = 0.5 * rep + 0.3 * (1.0 - variability) + 0.2 * (1.0 - sel);
        return clamp01(score);
    }

    double forestAggregate(double baseScore, double fScore, double cnnScore, double recScore, double rnnScore) const
    {
        baseScore = clamp01(baseScore);
        fScore = clamp01(fScore);
        cnnScore = clamp01(cnnScore);
        recScore = clamp01(recScore);
        rnnScore = clamp01(rnnScore);

        double cnnRnnAgreement = 1.0 - std::abs(cnnScore - rnnScore);
        double textModelScore = 0.65 * cnnScore + 0.35 * rnnScore;
        textModelScore = clamp01(textModelScore);

        std::array<double, 4> aux = {fScore, textModelScore, recScore, baseScore};
        double mean = 0.0;
        for (double v : aux)
            mean += v;
        mean /= (double)aux.size();
        double var = 0.0;
        for (double v : aux)
        {
            double d = v - mean;
            var += d * d;
        }
        var /= (double)aux.size();
        double disagreement = clamp01(std::sqrt(std::max(0.0, var)) * 2.0);

        double wBase = 0.35;
        double wCopy = 0.20;
        double wText = 0.25 * (0.5 + 0.5 * cnnRnnAgreement);
        double wRec = 0.20;
        double wSum = wBase + wCopy + wText + wRec;

        double linear = (wBase * baseScore + wCopy * fScore + wText * textModelScore + wRec * recScore) / std::max(1e-9, wSum);
        double interaction = 0.25 * baseScore * recScore +
                             0.20 * baseScore * textModelScore +
                             0.15 * fScore * textModelScore +
                             0.10 * recScore * fScore;
        interaction = clamp01(interaction);

        double conservative = std::max({baseScore, 0.85 * textModelScore, 0.9 * recScore});
        double uncertaintyPenalty = 0.28 * disagreement;
        double score = 0.52 * linear + 0.30 * interaction + 0.18 * conservative - uncertaintyPenalty;
        return clamp01(score);
    }

    double evaluateCompositeScore(const std::string &memeId, const NodeStats &ns,
                                  double avgConn, double avgOut, double stdDeg, double stdOut)
    {
        double baseScore = evaluateMaliciousness(memeId, ns, avgConn, avgOut, stdDeg, stdOut);
        std::vector<std::string> tokens = runtime_ ? runtime_->getMemeWords(memeId) : std::vector<std::string>();
        if (tokens.empty())
            tokens = tokenize(memeId);
        if (tokens.empty())
            return baseScore;
        ensureRecommenderReady();
        double fScore = computeCopyFeatures(tokens);
        double cnnScore = computeCnnScore(tokens);
        double rnnScore = 0.0;
#ifdef HAVE_TORCH
        if (useTorchModels_)
        {
            cnnScore = torchCnnScore(tokens);
            rnnScore = torchRnnScore(tokens);
        }
#endif
        double recScore = computeRecommenderScore(tokens);
        double forest = forestAggregate(baseScore, fScore, cnnScore, recScore, rnnScore);
        return clamp01(0.6 * forest + 0.4 * baseScore);
    }

    double getAverageConnections()
    {
        if (!runtime_)
            return 0.0;
        auto points = runtime_->graph().getAllPoints();
        if (points.empty())
            return 0;
        double sum = 0;
        for (auto &id : points)
            sum += runtime_->graph().degreeOf(id);
        return sum / points.size();
    }

    std::string generateReason(const std::string &memeId, double score)
    {
        if (!runtime_)
            return "score=0.000 conn=0";
        int degree = runtime_->graph().degreeOf(memeId);
        std::ostringstream oss;
        oss << "score=" << std::fixed << std::setprecision(3) << score << " conn=" << degree;
        if (useTextCNN_)
            oss << " cnn=on";
        else
            oss << " cnn=off";
        if (useRecommender_)
            oss << " rec=on";
        else
            oss << " rec=off";
        return oss.str();
    }

    bool isolateMeme(const std::string &memeId, double score, const std::string &reason)
    {
        if (!runtime_)
            return false;
        int cut = runtime_->graph().removeOutgoingEdges(memeId);
        if (cut > 0)
        {
            stats_.isolated++;
            json item{{"memeID", memeId}, {"score", score}, {"reason", reason}, {"cut", cut}};
            stats_.lastIsolated.insert(stats_.lastIsolated.begin(), item);
            if (stats_.lastIsolated.size() > 20)
                stats_.lastIsolated.resize(20);
            auto &logger = LoggerCXX::instance();
            if (logger.enabled())
                logger.log(LoggerCXX::Type::MONITORING,
                           std::string("memebarrier blocked meme=") + memeId +
                               " score=" + std::to_string(score) +
                               " cut=" + std::to_string(cut) +
                               " reason=" + reason);
            return true;
        }
        return false;
    }

    void scanNetwork()
    {
        if (!runtime_)
            return;
        auto points = runtime_->graph().getAllPoints();
        stats_.scans++;
        stats_.lastScanTime = nowEpochMs();
        if (points.empty())
            return;
        struct Scored
        {
            std::string id;
            double score;
            std::string reason;
        };
        std::vector<Scored> scored;
        scored.reserve(points.size());
        std::unordered_map<std::string, NodeStats> nodeStats;
        nodeStats.reserve(points.size());
        double sumDeg = 0.0;
        double sumOut = 0.0;
        double sumDeg2 = 0.0;
        double sumOut2 = 0.0;
        for (auto &id : points)
        {
            auto neighbors = runtime_->graph().neighborsOf(id);
            NodeStats ns;
            ns.degree = (int)neighbors.size();
            for (auto &n : neighbors)
            {
                if (n.first == id)
                    ns.selfLoops++;
                if (n.second.second == 2)
                    ns.outDegree++;
            }
            nodeStats.emplace(id, ns);
            sumDeg += ns.degree;
            sumOut += ns.outDegree;
            sumDeg2 += (double)ns.degree * ns.degree;
            sumOut2 += (double)ns.outDegree * ns.outDegree;
        }
        double avgDeg = sumDeg / points.size();
        double avgOut = sumOut / points.size();
        double varDeg = sumDeg2 / points.size() - avgDeg * avgDeg;
        double varOut = sumOut2 / points.size() - avgOut * avgOut;
        double stdDeg = std::sqrt(std::max(0.0, varDeg));
        double stdOut = std::sqrt(std::max(0.0, varOut));
        stats_.avgDegree = avgDeg;
        stats_.stdDegree = stdDeg;
        stats_.avgOutDegree = avgOut;
        stats_.stdOutDegree = stdOut;

        std::vector<double> scoreValues;
        scoreValues.reserve(points.size());
        for (auto &id : points)
        {
            const auto &ns = nodeStats[id];
            double s = evaluateCompositeScore(id, ns, avgDeg, avgOut, stdDeg, stdOut);
            scored.push_back(Scored{id, s, generateReason(id, s)});
            scoreValues.push_back(s);
        }

        std::sort(scoreValues.begin(), scoreValues.end());
        stats_.evaluated = (int)scoreValues.size();
        stats_.avgScore = 0.0;
        stats_.maxScore = scoreValues.empty() ? 0.0 : scoreValues.back();
        for (double v : scoreValues)
            stats_.avgScore += v;
        if (!scoreValues.empty())
            stats_.avgScore /= scoreValues.size();
        if (!scoreValues.empty())
        {
            size_t p90 = (size_t)std::floor(0.9 * (scoreValues.size() - 1));
            stats_.p90Score = scoreValues[p90];
        }
        else
        {
            stats_.p90Score = 0.0;
        }

        stats_.scoreHistogram.assign(10, 0);
        for (double v : scoreValues)
        {
            int bucket = (int)std::floor(std::min(0.999, std::max(0.0, v)) * 10.0);
            stats_.scoreHistogram[bucket]++;
        }

        double dynamicTarget = clamp01(0.7 * baseThreshold_ + 0.3 * stats_.p90Score);
        std::sort(scored.begin(), scored.end(), [](const Scored &a, const Scored &b)
                  { return a.score > b.score; });
        int isolated = 0;
        stats_.candidates = 0;
        for (auto &item : scored)
        {
            if (item.score >= maliciousThreshold_)
            {
                stats_.candidates++;
                auto &hits = hitCounts_[item.id];
                if (item.score >= maliciousThreshold_ + minIsolationMargin_)
                    hits = minConsecutiveHits_;
                else
                    hits = std::min(minConsecutiveHits_, hits + 1);
                if (hits >= minConsecutiveHits_)
                {
                    if (isolateMeme(item.id, item.score, item.reason))
                    {
                        isolated++;
                        if (isolated >= maxIsolatePerScan_)
                            break;
                    }
                }
            }
            else
            {
                hitCounts_[item.id] = 0;
                break;
            }
        }
        if (adaptiveEnabled_ && !scoreValues.empty())
        {
            double isolatedRate = stats_.evaluated > 0 ? (double)isolated / stats_.evaluated : 0.0;
            double high = std::max(0.002, disableRateTarget_ * 1.5);
            double low = std::max(0.0005, disableRateTarget_ * 0.5);
            if (isolatedRate > high)
                dynamicTarget = clamp01(dynamicTarget + 0.05);
            if (isolatedRate < low)
                dynamicTarget = clamp01(dynamicTarget - 0.02);
            double blended = maliciousThreshold_ * (1.0 - thresholdMomentum_) + dynamicTarget * thresholdMomentum_;
            maliciousThreshold_ = std::min(maxThreshold_, std::max(minThreshold_, blended));
        }
        stats_.lastThreshold = maliciousThreshold_;
    }
};

std::string OnlineResearcher::normalizeKey(const std::vector<std::string> &words) const
{
    std::ostringstream oss;
    int limit = std::min<int>(32, (int)words.size());
    for (int i = 0; i < limit; i++)
    {
        if (i)
            oss << " ";
        std::string w = words[i];
        for (auto &c : w)
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        oss << w;
    }
    return oss.str();
}

void OnlineResearcher::pruneCache()
{
    if ((int)cache_.size() <= cacheSize_)
        return;
    std::vector<std::pair<std::string, CacheEntry>> items(cache_.begin(), cache_.end());
    std::sort(items.begin(), items.end(), [](auto &a, auto &b)
              { return a.second.ts < b.second.ts; });
    while ((int)items.size() > cacheSize_)
    {
        cache_.erase(items.front().first);
        items.erase(items.begin());
    }
}

void OnlineResearcher::remember(const std::string &key, const json &payload)
{
    cache_[key] = CacheEntry{(int64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
                                 std::chrono::system_clock::now().time_since_epoch())
                                 .count(),
                             payload};
    pruneCache();
}

std::optional<json> OnlineResearcher::fromCache(const std::string &key)
{
    auto it = cache_.find(key);
    if (it == cache_.end())
        return std::nullopt;
    int64_t nowMs = (int64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::system_clock::now().time_since_epoch())
                        .count();
    if (nowMs - it->second.ts > cooldownMs_)
    {
        cache_.erase(it);
        return std::nullopt;
    }
    return it->second.payload;
}

std::vector<std::string> OnlineResearcher::loadSeedUrls(int maxSeeds) const
{
    std::vector<std::string> out;
    fs::path seeds = runtime_ ? (runtime_->config().robotsDir / "seeds.txt") : (getProjectRoot() / "robots" / "seeds.txt");
    if (!fs::exists(seeds))
        return out;
    std::ifstream in(seeds);
    if (!in)
        return out;
    std::string line;
    while (std::getline(in, line))
    {
        line = trimCopy(line);
        if (line.empty())
            continue;
        if (line[0] == '#')
            continue;
        out.push_back(line);
        if (maxSeeds > 0 && (int)out.size() >= maxSeeds)
            break;
    }
    return out;
}

std::vector<std::string> OnlineResearcher::extractLinksFromHtml(const std::string &html, const std::string &baseUrl) const
{
    std::vector<std::string> links;
    if (html.empty())
        return links;
    std::unordered_set<std::string> uniq;
    auto makeAbsolute = [&](const std::string &link) -> std::string
    {
        if (link.empty())
            return {};
        if (link.rfind("http://", 0) == 0 || link.rfind("https://", 0) == 0)
            return link;
        if (link.rfind("//", 0) == 0)
            return std::string("https:") + link;
        if (baseUrl.empty())
            return link;
        auto schemePos = baseUrl.find("://");
        auto pathPos = schemePos == std::string::npos ? std::string::npos : baseUrl.find('/', schemePos + 3);
        std::string root = (pathPos == std::string::npos) ? baseUrl : baseUrl.substr(0, pathPos);
        if (!link.empty() && link[0] == '/')
            return root + link;
        std::string baseDir = baseUrl;
        auto lastSlash = baseDir.find_last_of('/');
        if (lastSlash != std::string::npos)
            baseDir = baseDir.substr(0, lastSlash + 1);
        return baseDir + link;
    };
#ifdef HAVE_GUMBO
    GumboOutput *output = gumbo_parse(html.c_str());
    if (output)
    {
        std::function<void(GumboNode *)> walk = [&](GumboNode *node)
        {
            if (!node)
                return;
            if (node->type == GUMBO_NODE_ELEMENT)
            {
                GumboElement &el = node->v.element;
                auto addAttr = [&](const char *attrName)
                {
                    GumboAttribute *attr = gumbo_get_attribute(&el.attributes, attrName);
                    if (!attr || !attr->value)
                        return;
                    std::string link = attr->value;
                    if (link.rfind("javascript:", 0) == 0 || link.rfind("mailto:", 0) == 0)
                        return;
                    link = makeAbsolute(link);
                    link = normalizeUrlSimple(link);
                    if (link.empty())
                        return;
                    if (uniq.insert(link).second)
                        links.push_back(link);
                };
                addAttr("href");
                addAttr("src");
                GumboVector *children = &el.children;
                for (size_t i = 0; i < children->length; i++)
                {
                    walk(static_cast<GumboNode *>(children->data[i]));
                }
            }
            else if (node->type == GUMBO_NODE_DOCUMENT)
            {
                GumboVector *children = &node->v.document.children;
                for (size_t i = 0; i < children->length; i++)
                {
                    walk(static_cast<GumboNode *>(children->data[i]));
                }
            }
        };
        walk(output->root);
        gumbo_destroy_output(&kGumboDefaultOptions, output);
    }
#else
    std::regex hrefRe(R"(\b(?:href|src)\s*=\s*['"]([^'"#]+)['"])", std::regex::icase);
    std::smatch m;
    std::string::const_iterator searchStart(html.cbegin());
    while (std::regex_search(searchStart, html.cend(), m, hrefRe))
    {
        std::string link = m[1].str();
        if (link.rfind("javascript:", 0) == 0 || link.rfind("mailto:", 0) == 0)
        {
            searchStart = m.suffix().first;
            continue;
        }
        link = makeAbsolute(link);
        link = normalizeUrlSimple(link);
        if (!link.empty() && uniq.insert(link).second)
            links.push_back(link);
        searchStart = m.suffix().first;
    }
#endif
    return links;
}

std::string OnlineResearcher::cleanTextAdvanced(const std::string &text) const
{
    if (text.empty())
        return {};
    auto decodeEntities = [](const std::string &s)
    {
        std::string out = s;
        auto replaceAll = [&](const std::string &from, const std::string &to)
        {
            size_t pos = 0;
            while ((pos = out.find(from, pos)) != std::string::npos)
            {
                out.replace(pos, from.size(), to);
                pos += to.size();
            }
        };
        replaceAll("&nbsp;", " ");
        replaceAll("&amp;", "&");
        replaceAll("&lt;", "<");
        replaceAll("&gt;", ">");
        replaceAll("&quot;", "\"");
        replaceAll("&#39;", "'");
        return out;
    };
    std::string raw = decodeEntities(text);
    std::string cleaned;
    cleaned.reserve(raw.size());
    for (char c : raw)
    {
        unsigned char uc = static_cast<unsigned char>(c);
        if (uc < 0x09)
            continue;
        if (uc == 0x0b || uc == 0x0c)
            continue;
        cleaned.push_back(c);
    }
    std::unordered_set<std::string> seenLines;
    std::ostringstream oss;
    std::istringstream iss(cleaned);
    std::string line;
    auto alphaRatio = [](const std::string &s)
    {
        if (s.empty())
            return 0.0;
        int alpha = 0;
        int total = 0;
        for (unsigned char c : s)
        {
            if (std::isalpha(c))
                alpha++;
            if (!std::isspace(c))
                total++;
        }
        if (total == 0)
            return 0.0;
        return (double)alpha / (double)total;
    };
    while (std::getline(iss, line))
    {
        line = trimCopy(line);
        if (line.empty())
            continue;
        if (line.size() < 15 && alphaRatio(line) < 0.4)
            continue;
        if (line.size() > 600)
            line = line.substr(0, 600);
        if (!seenLines.insert(line).second)
            continue;
        if (!oss.str().empty())
            oss << "\n";
        oss << line;
        if ((int)oss.str().size() > 50000)
            break;
    }
    std::string out = oss.str();
    if (out.size() > 50000)
        out.resize(50000);
    return out;
}

std::vector<std::string> OnlineResearcher::pickAspects(const std::vector<std::string> &tokens, int maxAspects) const
{
    std::unordered_map<std::string, int> freq;
    auto &stop = getStopWords();
    for (const auto &t : tokens)
    {
        if (t.size() < 2)
            continue;
        if (stop.count(t))
            continue;
        freq[t] += 1;
    }
    std::vector<std::pair<std::string, int>> ranked(freq.begin(), freq.end());
    std::sort(ranked.begin(), ranked.end(), [](auto &a, auto &b)
              { return a.second > b.second; });
    std::vector<std::string> out;
    for (auto &p : ranked)
    {
        out.push_back(p.first);
        if ((int)out.size() >= maxAspects)
            break;
    }
    return out;
}

void OnlineResearcher::indexDocument(const DocEntry &doc)
{
    if (doc.url.empty())
        return;
    std::lock_guard<std::mutex> lock(indexMu_);
    docs_[doc.url] = doc;
    urlToContent_[doc.url] = doc.text;
    auto &stop = getStopWords();
    for (const auto &t : doc.tokens)
    {
        if (t.size() < 2)
            continue;
        if (stop.count(t))
            continue;
        invertedIndex_[t].insert(doc.url);
    }
    for (const auto &a : doc.aspects)
    {
        if (a.empty())
            continue;
        aspectToUrls_[a].insert(doc.url);
    }
}

json OnlineResearcher::queryLocalIndex(const std::vector<std::string> &words, int limitUrls, int limitAspects, int maxClusters)
{
    json out{{"ok", true}, {"source", "index"}, {"query", ""}};
    if (words.empty())
        return json{{"ok", false}, {"source", "index"}, {"reason", "empty-query"}};
    std::unordered_map<std::string, int> scores;
    std::vector<std::string> queryTokens = words;
    std::ostringstream oss;
    for (size_t i = 0; i < queryTokens.size(); i++)
    {
        if (i)
            oss << " ";
        oss << queryTokens[i];
    }
    out["query"] = oss.str();
    {
        std::lock_guard<std::mutex> lock(indexMu_);
        for (const auto &t : queryTokens)
        {
            auto it = invertedIndex_.find(t);
            if (it == invertedIndex_.end())
                continue;
            for (const auto &url : it->second)
                scores[url] += 1;
        }
    }
    std::vector<std::pair<std::string, int>> ranked(scores.begin(), scores.end());
    std::sort(ranked.begin(), ranked.end(), [](auto &a, auto &b)
              { return a.second > b.second; });
    json urls = json::array();
    std::vector<DocEntry> docs;
    {
        std::lock_guard<std::mutex> lock(indexMu_);
        for (auto &p : ranked)
        {
            auto it = docs_.find(p.first);
            if (it == docs_.end())
                continue;
            json item{{"url", it->second.url}, {"score", p.second}, {"title", it->second.title}};
            if (!it->second.aspects.empty())
                item["aspects"] = it->second.aspects;
            urls.push_back(item);
            docs.push_back(it->second);
            if ((int)urls.size() >= limitUrls)
                break;
        }
    }
    out["urls"] = urls;

    json aspects = json::array();
    {
        std::unordered_map<std::string, int> aspectScore;
        std::lock_guard<std::mutex> lock(indexMu_);
        for (const auto &t : queryTokens)
        {
            auto it = aspectToUrls_.find(t);
            if (it != aspectToUrls_.end())
            {
                aspectScore[t] = (int)it->second.size();
            }
        }
        std::vector<std::pair<std::string, int>> rankedA(aspectScore.begin(), aspectScore.end());
        std::sort(rankedA.begin(), rankedA.end(), [](auto &a, auto &b)
                  { return a.second > b.second; });
        for (auto &p : rankedA)
        {
            aspects.push_back(json{{"aspect", p.first}, {"count", p.second}});
            if ((int)aspects.size() >= limitAspects)
                break;
        }
    }
    out["aspects"] = aspects;

    if (!docs.empty() && maxClusters > 0)
    {
        std::vector<std::vector<float>> vectors;
        vectors.reserve(docs.size());
        for (auto &d : docs)
            vectors.push_back(d.embedding);
        int k = std::min<int>(maxClusters, std::max(1, (int)std::sqrt((double)vectors.size())));
        std::vector<int> assignments;
        auto centers = kMeans(vectors, k, 8, assignments);
        json clusters = json::array();
        for (int c = 0; c < k; c++)
        {
            json group = json::array();
            for (size_t i = 0; i < assignments.size(); i++)
            {
                if (assignments[i] == c)
                {
                    group.push_back(json{{"url", docs[i].url}, {"title", docs[i].title}});
                }
            }
            if (!group.empty())
                clusters.push_back(json{{"cluster", c}, {"size", (int)group.size()}, {"urls", group}});
        }
        out["clusters"] = clusters;
    }
    return out;
}

std::vector<std::vector<float>> OnlineResearcher::kMeans(const std::vector<std::vector<float>> &vectors, int k, int iters, std::vector<int> &assignments)
{
    std::vector<std::vector<float>> centers;
    assignments.assign(vectors.size(), 0);
    if (vectors.empty() || k <= 0)
        return centers;
    int dim = (int)vectors[0].size();
    centers.resize(k, std::vector<float>(dim, 0.0f));
    for (int i = 0; i < k; i++)
    {
        centers[i] = vectors[(size_t)i % vectors.size()];
    }
    auto dist2 = [&](const std::vector<float> &a, const std::vector<float> &b)
    {
        double d = 0.0;
        size_t n = std::min(a.size(), b.size());
        for (size_t i = 0; i < n; i++)
        {
            double diff = (double)a[i] - (double)b[i];
            d += diff * diff;
        }
        return d;
    };
    for (int it = 0; it < iters; it++)
    {
        for (size_t i = 0; i < vectors.size(); i++)
        {
            double best = std::numeric_limits<double>::max();
            int bestK = 0;
            for (int c = 0; c < k; c++)
            {
                double d = dist2(vectors[i], centers[c]);
                if (d < best)
                {
                    best = d;
                    bestK = c;
                }
            }
            assignments[i] = bestK;
        }
        std::vector<std::vector<float>> next(k, std::vector<float>(dim, 0.0f));
        std::vector<int> counts(k, 0);
        for (size_t i = 0; i < vectors.size(); i++)
        {
            int c = assignments[i];
            counts[c]++;
            for (int d = 0; d < dim; d++)
                next[c][d] += vectors[i][d];
        }
        for (int c = 0; c < k; c++)
        {
            if (counts[c] == 0)
                continue;
            for (int d = 0; d < dim; d++)
                next[c][d] /= (float)counts[c];
        }
        centers.swap(next);
    }
    return centers;
}

json OnlineResearcher::fallback(const std::vector<std::string> &words)
{
    if (!runtime_)
        return json{{"ok", false}, {"source", "local"}, {"reason", "runtime-missing"}};
    if (words.empty())
        return json{{"ok", false}, {"source", "local"}, {"reason", "empty-query"}};
    auto mapping = runtime_->mapWordsToMemesOrdered(words);
    auto seeds = mapping.seeds;
    if (seeds.empty())
    {
        return json{{"ok", false}, {"source", "local"}, {"reason", "no-memes"}};
    }
    auto result = runtime_->runPropagation(seeds, json{{"seedOrder", mapping.order}});
    json suggestions = json::array();
    int limit = std::min<int>(10, (int)result.windowInfo.ids.size());
    for (int i = 0; i < limit; i++)
    {
        std::string memeId = result.windowInfo.ids[i];
        double strength = (i < (int)result.activation.size() && std::isfinite(result.activation[i])) ? result.activation[i] : 0.0;
        std::vector<std::string> wordsForMeme;
        for (auto &w : runtime_->kvm().getMemeWords(memeId))
        {
            wordsForMeme.push_back(w);
            if (wordsForMeme.size() >= 6)
                break;
        }
        suggestions.push_back(json{{"memeId", memeId}, {"strength", strength}, {"words", wordsForMeme}});
    }
    std::ostringstream oss;
    for (size_t i = 0; i < words.size(); i++)
    {
        if (i)
            oss << " ";
        oss << words[i];
    }
    return json{{"ok", true}, {"source", "local"}, {"query", oss.str()}, {"suggestions", suggestions}};
}

json OnlineResearcher::crawl(const std::string &startUrl, const json &crawlReq, const json &options)
{
    json out;
    out["ok"] = true;
    out["source"] = "crawl";
    out["pages"] = json::array();
#ifdef HAVE_CURL
    int timeoutMs = clampInt(crawlReq.value("timeoutMs", json()), 1000, 60000, 12000);
    int maxPages = clampInt(crawlReq.value("maxPages", json()), 1, 500, 60);
    int maxDepth = clampInt(crawlReq.value("maxDepth", json()), 0, 10, 3);
    int maxSeeds = clampInt(crawlReq.value("maxSeeds", json()), 0, 5000, 200);
    bool includePdf = crawlReq.value("includePdf", true);
    bool sameSite = crawlReq.value("sameSite", true);
    int maxBytesPerPage = clampInt(crawlReq.value("maxBytesPerPage", json()), 8 * 1024, 10 * 1024 * 1024, 2 * 1024 * 1024);
    int maxPdfBytes = clampInt(crawlReq.value("maxPdfBytes", json()), 64 * 1024, 40 * 1024 * 1024, 20 * 1024 * 1024);
    (void)maxPdfBytes;
    std::string userAgent = crawlReq.value("userAgent", "079ProjectCrawler/1.0");
    bool useSeeds = crawlReq.value("useSeeds", true);
    int kClusters = clampInt(crawlReq.value("kClusters", json()), 0, 12, 4);
    int maxAspects = clampInt(crawlReq.value("maxAspects", json()), 2, 30, 12);
    json errors = json::array();
    struct CrawlItem
    {
        std::string url;
        int depth{0};
        std::string root;
    };

    std::queue<CrawlItem> q;
    std::unordered_set<std::string> visited;
    std::unordered_set<std::string> enqueued;
    std::vector<std::string> roots;
    std::string normalizedStart = normalizeUrlSimple(startUrl);
    if (!normalizedStart.empty())
        roots.push_back(normalizedStart);
    if (useSeeds)
    {
        auto seeds = loadSeedUrls(maxSeeds);
        for (auto &s : seeds)
        {
            auto norm = normalizeUrlSimple(s);
            if (!norm.empty())
                roots.push_back(norm);
        }
    }
    for (auto &r : roots)
    {
        if (enqueued.insert(r).second)
            q.push({r, 0, r});
    }
    if (q.empty())
        return json{{"ok", false}, {"source", "crawl"}, {"reason", "startUrl-or-seeds-required"}};

    std::unordered_map<std::string, std::unordered_set<std::string>> localAspectToUrls;
    std::unordered_map<std::string, std::unordered_set<std::string>> localInvertedIndex;
    std::vector<DocEntry> localDocs;
    while (!q.empty() && (int)out["pages"].size() < maxPages)
    {
        auto item = q.front();
        q.pop();
        auto url = item.url;
        int depth = item.depth;
        if (url.empty() || visited.count(url))
            continue;
        visited.insert(url);
        if (sameSite && !item.root.empty() && !sameSiteSimple(item.root, url))
            continue;

        auto res = fetchUrl(url, timeoutMs, userAgent, (size_t)maxBytesPerPage);
        std::string text;
        std::vector<std::string> links;
#ifdef HAVE_GUMBO
        if (res.contentType.find("text/html") != std::string::npos || res.body.find("<html") != std::string::npos)
        {
            text = extractTextFromHtml(res.body);
            links = extractLinksFromHtml(res.body, url);
        }
#endif
#ifdef HAVE_POPPLER
        if (text.empty() && includePdf && (res.contentType.find("application/pdf") != std::string::npos || url.find(".pdf") != std::string::npos))
        {
            fs::path tmp = runtime_ ? (runtime_->config().baseDir / "_tmp_fetch.pdf") : fs::path("_tmp_fetch.pdf");
            std::ofstream ofs(tmp, std::ios::binary);
            ofs << res.body;
            ofs.close();
            text = extractTextFromPdfFile(tmp);
        }
#endif
        if (!text.empty())
        {
            std::string cleaned = cleanTextAdvanced(text);
            std::string clipped = cleaned.size() > 20000 ? cleaned.substr(0, 20000) : cleaned;
            std::vector<std::string> tokens = tokenize(cleaned);
            std::vector<std::string> aspects = pickAspects(tokens, maxAspects);
            DocEntry doc;
            doc.url = url;
            doc.text = cleaned;
            doc.tokens = tokens;
            doc.aspects = aspects;
            doc.embedding = textToMiniEmbedding(cleaned, 64);
            indexDocument(doc);
            localDocs.push_back(doc);
            for (const auto &t : tokens)
            {
                if (t.size() < 2 || getStopWords().count(t))
                    continue;
                localInvertedIndex[t].insert(url);
            }
            for (const auto &a : aspects)
            {
                localAspectToUrls[a].insert(url);
            }
            out["pages"].push_back(json{{"url", url}, {"status", res.status}, {"contentType", res.contentType}, {"bytes", (int64_t)res.bytes}, {"linksFound", (int)links.size()}, {"text", clipped}, {"aspects", aspects}});
        }
        else if (res.status <= 0)
        {
            errors.push_back(json{{"url", url}, {"error", "fetch-failed"}});
        }

        if (depth < maxDepth)
        {
            for (auto &link : links)
            {
                if (link.empty() || visited.count(link))
                    continue;
                if (sameSite && !item.root.empty() && !sameSiteSimple(item.root, link))
                    continue;
                if (enqueued.insert(link).second)
                    q.push({link, depth + 1, item.root});
            }
        }
    }

    json chunks = json::array();
    std::string aggregated;
    for (auto &p : out["pages"])
    {
        std::string url = p.value("url", "");
        std::string text = p.value("text", "");
        if (text.empty())
            continue;
        std::string block = "URL: " + url + "\n" + text;
        if (aggregated.size() + block.size() + 2 > 60000)
            break;
        if (!aggregated.empty())
            aggregated += "\n\n";
        aggregated += block;
    }
    out["startUrl"] = normalizedStart;
    out["stats"] = json{{"visited", (int)visited.size()}, {"returned", (int)out["pages"].size()}, {"queued", (int)q.size()}, {"errors", (int)errors.size()}};
    out["text"] = aggregated;
    json errOut = json::array();
    for (size_t i = 0; i < errors.size() && i < 20; i++)
        errOut.push_back(errors[i]);
    out["errors"] = errOut;

    json aspectMap = json::array();
    int aspectCount = 0;
    for (auto &kv : localAspectToUrls)
    {
        json urls = json::array();
        int take = 0;
        for (auto &u : kv.second)
        {
            urls.push_back(u);
            if (++take >= 5)
                break;
        }
        aspectMap.push_back(json{{"aspect", kv.first}, {"urls", urls}});
        if (++aspectCount >= 24)
            break;
    }
    out["aspectUrlMap"] = aspectMap;

    json inv = json::array();
    int invCount = 0;
    for (auto &kv : localInvertedIndex)
    {
        inv.push_back(json{{"term", kv.first}, {"count", (int)kv.second.size()}});
        if (++invCount >= 24)
            break;
    }
    out["invertedIndex"] = inv;

    if (!localDocs.empty() && kClusters > 0)
    {
        std::vector<std::vector<float>> vectors;
        for (auto &d : localDocs)
            vectors.push_back(d.embedding);
        int k = std::min<int>(kClusters, std::max(1, (int)std::sqrt((double)vectors.size())));
        std::vector<int> assignments;
        auto centers = kMeans(vectors, k, 8, assignments);
        (void)centers;
        json clusters = json::array();
        for (int c = 0; c < k; c++)
        {
            json urls = json::array();
            for (size_t i = 0; i < assignments.size(); i++)
            {
                if (assignments[i] == c)
                    urls.push_back(localDocs[i].url);
            }
            if (!urls.empty())
                clusters.push_back(json{{"cluster", c}, {"size", (int)urls.size()}, {"urls", urls}});
        }
        out["clusters"] = clusters;
    }
#endif
    (void)options;
    return out;
}
#if 0
    struct CrawlItem
    {
        std::string url;
        int depth{0};
        std::string root;
    };
#endif
#if 0
    std::queue<CrawlItem> q;
    std::unordered_set<std::string> visited;
    std::unordered_set<std::string> enqueued;
    std::vector<std::string> roots;
    std::string normalizedStart = normalizeUrlSimple(startUrl);
    if (!normalizedStart.empty())
        roots.push_back(normalizedStart);
    if (useSeeds)
    {
        auto seeds = loadSeedUrls(maxSeeds);
        for (auto &s : seeds)
        {
            auto norm = normalizeUrlSimple(s);
            if (!norm.empty())
                roots.push_back(norm);
        }
    }
    for (auto &r : roots)
    {
        if (enqueued.insert(r).second)
            q.push({r, 0, r});
    }
    if (q.empty())
        return json{{"ok", false}, {"source", "crawl"}, {"reason", "startUrl-or-seeds-required"}};

    std::unordered_map<std::string, std::unordered_set<std::string>> localAspectToUrls;
    std::unordered_map<std::string, std::unordered_set<std::string>> localInvertedIndex;
    std::vector<DocEntry> localDocs;
    while (!q.empty() && (int)out["pages"].size() < maxPages)
    {
        auto item = q.front();
        q.pop();
        auto url = item.url;
        int depth = item.depth;
        if (url.empty() || visited.count(url))
            continue;
        visited.insert(url);
        if (sameSite && !item.root.empty() && !sameSiteSimple(item.root, url))
            continue;

        auto res = fetchUrl(url, timeoutMs, userAgent, (size_t)maxBytesPerPage);
        std::string text;
        std::vector<std::string> links;
#ifdef HAVE_GUMBO
        if (res.contentType.find("text/html") != std::string::npos || res.body.find("<html") != std::string::npos)
        {
            text = extractTextFromHtml(res.body);
            links = extractLinksFromHtml(res.body, url);
        }
#endif
#ifdef HAVE_POPPLER
        if (text.empty() && includePdf && (res.contentType.find("application/pdf") != std::string::npos || url.find(".pdf") != std::string::npos))
        {
            fs::path tmp = runtime_ ? (runtime_->config().baseDir / "_tmp_fetch.pdf") : fs::path("_tmp_fetch.pdf");
            std::ofstream ofs(tmp, std::ios::binary);
            ofs << res.body;
            ofs.close();
            text = extractTextFromPdfFile(tmp);
        }
#endif
        if (!text.empty())
        {
            std::string cleaned = cleanTextAdvanced(text);
            std::string clipped = cleaned.size() > 20000 ? cleaned.substr(0, 20000) : cleaned;
            std::vector<std::string> tokens = tokenize(cleaned);
            std::vector<std::string> aspects = pickAspects(tokens, maxAspects);
            DocEntry doc;
            doc.url = url;
            doc.text = cleaned;
            doc.tokens = tokens;
            doc.aspects = aspects;
            doc.embedding = textToMiniEmbedding(cleaned, 64);
            indexDocument(doc);
            localDocs.push_back(doc);
            for (const auto &t : tokens)
            {
                if (t.size() < 2 || getStopWords().count(t))
                    continue;
                localInvertedIndex[t].insert(url);
            }
            for (const auto &a : aspects)
            {
                localAspectToUrls[a].insert(url);
            }
            out["pages"].push_back(json{{"url", url}, {"status", res.status}, {"contentType", res.contentType}, {"bytes", (int64_t)res.bytes}, {"linksFound", (int)links.size()}, {"text", clipped}, {"aspects", aspects}});
        }
        else if (res.status <= 0)
        {
            errors.push_back(json{{"url", url}, {"error", "fetch-failed"}});
        }

        if (depth < maxDepth)
        {
            for (auto &link : links)
            {
                if (link.empty() || visited.count(link))
                    continue;
                if (sameSite && !item.root.empty() && !sameSiteSimple(item.root, link))
                    continue;
                if (enqueued.insert(link).second)
                    q.push({link, depth + 1, item.root});
            }
        }
    }

    json chunks = json::array();
    std::string aggregated;
    for (auto &p : out["pages"])
    {
        std::string url = p.value("url", "");
        std::string text = p.value("text", "");
        if (text.empty())
            continue;
        std::string block = "URL: " + url + "\n" + text;
        if (aggregated.size() + block.size() + 2 > 60000)
            break;
        if (!aggregated.empty())
            aggregated += "\n\n";
        aggregated += block;
    }
    out["startUrl"] = normalizedStart;
    out["stats"] = json{{"visited", (int)visited.size()}, {"returned", (int)out["pages"].size()}, {"queued", (int)q.size()}, {"errors", (int)errors.size()}};
    out["text"] = aggregated;
    json errOut = json::array();
    for (size_t i = 0; i < errors.size() && i < 20; i++)
        errOut.push_back(errors[i]);
    out["errors"] = errOut;

    json aspectMap = json::array();
    int aspectCount = 0;
    for (auto &kv : localAspectToUrls)
    {
        json urls = json::array();
        int take = 0;
        for (auto &u : kv.second)
        {
            urls.push_back(u);
            if (++take >= 5)
                break;
        }
        aspectMap.push_back(json{{"aspect", kv.first}, {"urls", urls}});
        if (++aspectCount >= 24)
            break;
    }
    out["aspectUrlMap"] = aspectMap;

    json inv = json::array();
    int invCount = 0;
    for (auto &kv : localInvertedIndex)
    {
        inv.push_back(json{{"term", kv.first}, {"count", (int)kv.second.size()}});
        if (++invCount >= 24)
            break;
    }
    out["invertedIndex"] = inv;

    if (!localDocs.empty() && kClusters > 0)
    {
        std::vector<std::vector<float>> vectors;
        for (auto &d : localDocs)
            vectors.push_back(d.embedding);
        int k = std::min<int>(kClusters, std::max(1, (int)std::sqrt((double)vectors.size())));
        std::vector<int> assignments;
        auto centers = kMeans(vectors, k, 8, assignments);
        (void)centers;
        json clusters = json::array();
        for (int c = 0; c < k; c++)
        {
            json urls = json::array();
            for (size_t i = 0; i < assignments.size(); i++)
            {
                if (assignments[i] == c)
                    urls.push_back(localDocs[i].url);
            }
            if (!urls.empty())
                clusters.push_back(json{{"cluster", c}, {"size", (int)urls.size()}, {"urls", urls}});
        }
        out["clusters"] = clusters;
    }
// #endif moved into 054_section_before_crawlitem.inc
    (void)options;
    return out;
}
#endif

json OnlineResearcher::lookup(const json &input, const json &options)
{
    std::string rawText;
    std::vector<std::string> words;
    std::function<std::string(const json &)> jsStringForJoin;
    jsStringForJoin = [&](const json &v) -> std::string
    {
        if (v.is_null())
            return "";
        if (v.is_string())
            return v.get<std::string>();
        if (v.is_boolean())
            return v.get<bool>() ? "true" : "false";
        if (v.is_number())
            return v.dump();
        if (v.is_array())
        {
            std::string out;
            bool first = true;
            for (auto &item : v)
            {
                if (!first)
                    out += ",";
                first = false;
                out += jsStringForJoin(item);
            }
            return out;
        }
        return "[object Object]";
    };
    auto jsStringOrEmpty = [&](const json &v) -> std::string
    {
        if (v.is_null())
            return "";
        if (v.is_boolean())
            return v.get<bool>() ? "true" : "";
        if (v.is_number())
        {
            double n = v.get<double>();
            if (n == 0.0)
                return "";
            return v.dump();
        }
        if (v.is_string())
            return v.get<std::string>();
        if (v.is_array())
        {
            std::string out;
            bool first = true;
            for (auto &item : v)
            {
                if (!first)
                    out += ",";
                first = false;
                out += jsStringForJoin(item);
            }
            return out;
        }
        return "[object Object]";
    };

    if (input.is_array())
    {
        for (auto &t : input)
        {
            words.push_back(jsStringForJoin(t));
        }
        for (size_t i = 0; i < words.size(); i++)
        {
            if (i)
                rawText += " ";
            rawText += words[i];
        }
    }
    else
    {
        rawText = jsStringOrEmpty(input);
        words = tokenize(rawText);
    }

    bool useIndex = options.value("useIndex", false) || options.value("mode", "search") == "aspect";
    if (useIndex)
    {
        auto idx = queryLocalIndex(words, options.value("limitUrls", 12), options.value("limitAspects", 10), options.value("kClusters", 4));
        if (options.value("preferIndex", true) || options.value("mode", "search") == "aspect")
        {
            return idx;
        }
    }

    if (!enabled_ || options.value("disableRemote", false))
    {
        return fallback(words);
    }

    auto crawlReq = options.value("crawl", json::object());
    std::string urlFromInput = extractFirstUrl(rawText);
    bool shouldCrawl = options.value("mode", "search") == "crawl" || options.value("mode", "search") == "seed" || (!urlFromInput.empty()) || (crawlReq.contains("startUrl") && crawlReq["startUrl"].is_string());
    if (shouldCrawl)
    {
        std::string startUrl = crawlReq.value("startUrl", urlFromInput);
        if (startUrl.empty() && options.value("mode", "search") != "seed")
        {
            return json{{"ok", false}, {"source", "crawl"}, {"reason", "startUrl-required"}};
        }
        auto payload = crawl(startUrl, crawlReq, options);
        if (!options.value("forceRemote", false))
        {
            std::string cacheSeed = startUrl.empty() ? "seed" : startUrl;
            auto key = normalizeKey(tokenize("crawl " + cacheSeed));
            remember(key, payload);
        }
        return payload;
    }

    if (words.empty())
    {
        return json{{"ok", false}, {"source", "local"}, {"reason", "empty-query"}};
    }
    std::string key = normalizeKey(words);
    if (!options.value("forceRemote", false))
    {
        auto cached = fromCache(key);
        if (cached)
        {
            json payload = *cached;
            payload["cached"] = true;
            return payload;
        }
    }

    json payload;
    bool remoteOk = false;
#ifdef HAVE_CURL
    if (!endpoint_.empty() && !options.value("skipRemote", false))
    {
        std::string url = endpoint_;
        if (url.find('?') == std::string::npos)
            url += "?q=" + urlEncode(rawText);
        else
            url += "&q=" + urlEncode(rawText);
        int timeoutMs = options.value("timeout", 5000);
        auto res = fetchUrl(url, timeoutMs);
        if (!res.body.empty())
        {
            json raw;
            try
            {
                raw = json::parse(res.body);
            }
            catch (...)
            {
                raw = json{{"raw", res.body}};
            }
            std::string snippet;
            if (raw.is_string())
                snippet = raw.get<std::string>().substr(0, 280);
            else
                snippet = raw.dump().substr(0, 280);
            payload = json{{"ok", true}, {"source", "remote"}, {"query", rawText}, {"snippet", snippet}};
            if (options.value("includeRaw", false))
                payload["raw"] = raw;
            remoteOk = true;
        }
    }
#endif
    if (!remoteOk)
    {
        payload = fallback(words);
    }

    remember(key, payload);
    return payload;
}

// ------------------ Controller / Pool ------------------

class Controller;

// ControllerPoolBase 定义控制器池抽象接口。
// 调用方式：网关与学习模块依赖该接口屏蔽实现差异。
// 实现思路：统一控制器发现、调用、指标与运维操作。
// 注意事项：不同实现的时延与一致性语义可能不同。
// 注意事项：调用方应处理空控制器与异常返回。
// 注意事项：该接口不约束具体线程模型。
class ControllerPoolBase
{
public:
    virtual ~ControllerPoolBase() = default;
    virtual std::shared_ptr<Controller> getActive() = 0;
    virtual std::shared_ptr<Controller> getStandby() = 0;
    virtual std::shared_ptr<Controller> getValidation() = 0;
    virtual std::shared_ptr<Controller> getByName(const std::string &name) = 0;
    virtual std::vector<std::string> listControllerNames() const = 0;
    virtual std::vector<std::string> listGroupIds() const = 0;
    virtual std::vector<std::string> listControllersInGroup(const std::string &gid) = 0;
    virtual std::vector<json> listMetrics() const = 0;
    virtual json ingestDocument(const json &doc) = 0;
    virtual json ingestDocumentToGroup(const std::string &gid, const json &doc) = 0;
    virtual json forgetMemes(const json &payload) = 0;
    virtual json onlineResearch(const json &queryOrTokens, const json &options) = 0;
    virtual void hotSwap(const json &snapshot) = 0;
    virtual const Config &config() const = 0;
    virtual json respondByName(const std::string &controllerName, const json &payload) = 0;
};

// Controller 抽象单个 AI 控制器的行为契约。
// 调用方式：池化层通过该接口进行统一调度与管理。
// 实现思路：覆盖响应、参数、快照与度量等关键能力。
// 注意事项：实现类需保证线程安全与异常隔离。
// 注意事项：snapshot/applySnapshot 需尽量幂等。
// 注意事项：name 必须稳定用于路由与指标汇聚。
class Controller
{
public:
    virtual ~Controller() = default;
    virtual json respond(const json &payload) = 0;
    virtual json metrics() const = 0;
    virtual void applyParams(const json &p) = 0;
    virtual json snapshot() const = 0;
    virtual void applySnapshot(const json &snap) = 0;
    virtual const std::string &name() const = 0;
    virtual std::shared_ptr<RuntimeBase> runtime() const = 0;
};

// ShardDescriptor 保存分片节点的健康与向量状态。
// 调用方式：由分片管理器在请求前后更新统计信息。
// 实现思路：通过 EMA 延迟与失败率估算可用性。
// 注意事项：embedding 维度应与路由向量保持一致。
// 注意事项：离线判定阈值影响流量稳定性。
// 注意事项：history 仅用于观测，不作为强一致日志。
class ShardDescriptor
{
public:
    explicit ShardDescriptor(std::string name)
        : name_(std::move(name)), label_("AI-" + name_), embedding_(64, 0.0f) {}

    const std::string &name() const { return name_; }
    const std::string &label() const { return label_; }
    const std::vector<float> &embedding() const { return embedding_; }
    bool isOffline() const { return offline_; }

    void updateEmbedding(const std::vector<float> &vec)
    {
        double alpha = 0.2;
        if (vec.size() != embedding_.size())
            return;
        for (size_t i = 0; i < embedding_.size(); i++)
        {
            embedding_[i] = (float)((1 - alpha) * embedding_[i] + alpha * vec[i]);
        }
    }

    void record(const json &entry)
    {
        requests_++;
        bool ok = entry.value("ok", true);
        if (!ok)
            failures_++;
        int latency = entry.value("latency", 0);
        if (latency > 0)
        {
            if (emaLatencyMs_ <= 0.0)
                emaLatencyMs_ = latency;
            else
                emaLatencyMs_ = 0.9 * emaLatencyMs_ + 0.1 * latency;
        }
        lastSeenMs_ = nowEpochMs();
        double failRate = requests_ > 0 ? (double)failures_ / requests_ : 0.0;
        offline_ = failRate > 0.45 || (emaLatencyMs_ > 3500.0 && requests_ > 6);

        history_.push_back(entry);
        if (history_.size() > 50)
            history_.erase(history_.begin());
    }

    json toJson() const
    {
        return json{{"name", name_},
                    {"label", label_},
                    {"offline", offline_},
                    {"emaLatencyMs", emaLatencyMs_},
                    {"requests", requests_},
                    {"failures", failures_},
                    {"lastSeenMs", lastSeenMs_},
                    {"history", history_}};
    }

private:
    std::string name_;
    std::string label_;
    std::vector<float> embedding_;
    std::vector<json> history_;
    bool offline_{false};
    int requests_{0};
    int failures_{0};
    double emaLatencyMs_{0.0};
    int64_t lastSeenMs_{0};
};

class ShardManager
{
public:
    explicit ShardManager(std::shared_ptr<ControllerPoolBase> pool) : pool_(std::move(pool))
    {
        auto names = pool_->listControllerNames();
        for (auto &name : names)
        {
            shards_.emplace(name, ShardDescriptor(name));
        }
        lastEmbedding_ = std::vector<float>(64, 0.0f);
    }

    std::shared_ptr<Controller> chooseShard(const std::string &text)
    {
        auto requestEmbedding = textToMiniEmbedding(text, 64);
        lastEmbedding_ = requestEmbedding;
        auto best = pool_->getActive();
        double bestScore = -1e9;
        for (auto &kv : shards_)
        {
            if (kv.second.isOffline())
                continue;
            auto score = cosineSim(requestEmbedding, kv.second.embedding());
            if (score > bestScore)
            {
                auto ctrl = pool_->getByName(kv.first);
                if (ctrl)
                {
                    bestScore = score;
                    best = ctrl;
                }
            }
        }
        if (!best)
            best = pool_->getActive();
        return best;
    }

    json metrics() const
    {
        json out = json::array();
        for (auto &kv : shards_)
        {
            out.push_back({{"name", kv.first}, {"shard", kv.second.toJson()}});
        }
        return out;
    }

    void record(const std::string &controllerName, const std::vector<float> &requestEmbedding, const std::string &replyText, int latency)
    {
        auto it = shards_.find(controllerName);
        if (it == shards_.end())
            return;
        auto replyEmbedding = textToMiniEmbedding(replyText, (int)requestEmbedding.size());
        it->second.updateEmbedding(replyEmbedding);
        double affinity = cosineSim(requestEmbedding, replyEmbedding);
        it->second.record(json{{"ok", true}, {"latency", latency}, {"affinity", affinity}, {"ts", (int64_t)std::chrono::system_clock::now().time_since_epoch().count()}});
    }

    void recordFailure(const std::string &controllerName, int latency = 0)
    {
        auto it = shards_.find(controllerName);
        if (it == shards_.end())
            return;
        it->second.record(json{{"ok", false}, {"latency", latency}, {"affinity", 0.0}, {"ts", (int64_t)std::chrono::system_clock::now().time_since_epoch().count()}});
    }

private:
    std::shared_ptr<ControllerPoolBase> pool_;
    std::unordered_map<std::string, ShardDescriptor> shards_;
    std::vector<float> lastEmbedding_;
};

class ControllerPool;

class InferenceProcessPool
{
public:
    InferenceProcessPool(ControllerPool *pool, int workers, int timeoutMs);
    ~InferenceProcessPool();

    json respond(const std::string &controllerName, const json &payload);
    void stop();

private:
    struct Task
    {
        std::string controllerName;
        SharedMemorySlice payload;
        std::shared_ptr<std::promise<SharedMemorySlice>> promise;
    };

    ControllerPool *pool_{nullptr};
    int timeoutMs_{12000};
    size_t maxQueue_{1024};
    std::atomic<bool> running_{false};
    std::vector<std::thread> workers_;
    std::queue<std::shared_ptr<Task>> queue_;
    std::mutex mu_;
    std::condition_variable cv_;

    void workerLoop();
};

class GroupProcessPool;

class ProxyRuntime : public RuntimeBase
{
public:
    ProxyRuntime(GroupProcessPool *pool, std::string controllerName)
        : pool_(pool), controllerName_(std::move(controllerName)) {}

    json processInput(const json &payload) override;
    json cloneParams() const override;
    void setParams(const json &patch) override;
    json toSnapshot() override;
    void fromSnapshot(const json &snapshot) override;
    json ingestDocument(const json &doc) override;
    json forgetMemes(const json &payload) override;
    json onlineLookup(const std::string &query, const json &options) override;
    json onlineLookup(const json &input, const json &options) override;
    json getSearchConfig() const override;
    json setSearchConfig(const json &patch) override;
    std::vector<RobotsDocument> collectRobotsDocuments(const json &options) const override;
    std::vector<std::string> listRobotsFiles() const override;
    fs::path exportGraphToFile(const json &options) override;
    json learnFromDialog(const json &payload, const json &result) override;
    std::unordered_map<std::string, double> mapWordsToMemes(const std::vector<std::string> &words) override;
    std::vector<std::string> getMemeWords(const std::string &memeId) override;
    void graphLink(const std::string &fromId, const std::string &toId, double weight, int direction) override;
    const Config &config() const override;

private:
    GroupProcessPool *pool_{nullptr};
    std::string controllerName_;
};

class ProxyController : public Controller
{
public:
    ProxyController(GroupProcessPool *pool, std::string name)
        : pool_(pool), name_(std::move(name)), runtime_(std::make_shared<ProxyRuntime>(pool, name_)) {}

    json respond(const json &payload) override;
    json metrics() const override;
    void applyParams(const json &p) override;
    json snapshot() const override;
    void applySnapshot(const json &snap) override;
    const std::string &name() const override { return name_; }
    std::shared_ptr<RuntimeBase> runtime() const override { return runtime_; }

private:
    GroupProcessPool *pool_{nullptr};
    std::string name_;
    std::shared_ptr<ProxyRuntime> runtime_;
};

class LocalController : public Controller
{
public:
    LocalController(std::string name, std::shared_ptr<RuntimeState> runtime)
        : name_(std::move(name)), runtime_(std::move(runtime)), since_(nowEpochMs()) {}

    json respond(const json &payload) override { return runtime_->processInput(payload); }
    json metrics() const override
    {
        return json{{"name", name_}, {"online", true}, {"health", json{{"status", "ok"}, {"since", since_}, {"failures", 0}}}, {"metrics", runtime_->metrics()}, {"params", runtime_->cloneParams()}};
    }
    void applyParams(const json &p) override { runtime_->setParams(p); }
    json snapshot() const override { return runtime_->toSnapshot(); }
    void applySnapshot(const json &snap) override { runtime_->fromSnapshot(snap); }
    const std::string &name() const override { return name_; }
    std::shared_ptr<RuntimeBase> runtime() const override { return runtime_; }

private:
    std::string name_;
    std::shared_ptr<RuntimeState> runtime_;
    int64_t since_{0};
};

class ControllerPool : public ControllerPoolBase
{
public:
    ControllerPool(std::shared_ptr<KeyValueStore> kvmStore,
                   std::shared_ptr<KeyValueStore> memeStore,
                   std::shared_ptr<KeyValueStore> sessionStore,
                   const Config &config)
        : config_(config)
    {
#ifdef HAVE_LMDB
        bool preferSqlite = Database079::isSqliteStore(kvmStore) || Database079::isSqliteStore(memeStore) || Database079::isSqliteStore(sessionStore);
        if (!preferSqlite)
        {
            auto lmdbK = std::make_shared<LmdbStore>("kvm", config_.lmdbRoot, config_.lmdbMapSizeBytes);
            auto lmdbM = std::make_shared<LmdbStore>("meme_graph", config_.lmdbRoot, config_.lmdbMapSizeBytes);
            auto lmdbS = std::make_shared<LmdbStore>("session", config_.lmdbRoot, config_.lmdbMapSizeBytes);
            if (lmdbK->ok())
                kvmStore = lmdbK;
            if (lmdbM->ok())
                memeStore = lmdbM;
            if (lmdbS->ok())
                sessionStore = lmdbS;
        }
#endif
        for (int i = 0; i < config_.groupCount; i++)
        {
            std::string gid = "G" + std::to_string(i + 1);
            groupIds_.push_back(gid);
            std::vector<std::string> names;
            auto kvmNs = std::make_shared<NamespacedStore>(kvmStore, gid + ":kvm");
            auto memeNs = std::make_shared<NamespacedStore>(memeStore, gid + ":graph");
            auto sessNs = std::make_shared<NamespacedStore>(sessionStore, gid + ":session");
            for (int j = 0; j < config_.groupSize; j++)
            {
                std::string name = gid + "_AI" + std::to_string(j + 1);
                controllers_[name] = std::make_shared<LocalController>(name, std::make_shared<RuntimeState>(kvmNs, memeNs, sessNs, config_));
                names.push_back(name);
            }
            groups_[gid] = names;
        }
        auto g1 = groupIds_.empty() ? "G1" : groupIds_[0];
        auto list = groups_[g1];
        serving_ = controllers_[list[0]];
        standby_ = controllers_[list.size() > 1 ? list[1] : list[0]];
        validation_ = controllers_[list.size() > 2 ? list[2] : list[0]];
        if (config_.inferMP && !config_.groupProc)
        {
            inferencePool_ = std::make_unique<InferenceProcessPool>(this, config_.inferWorkers, config_.groupProcTimeoutMs);
        }
    }

    std::shared_ptr<Controller> getActive() override { return serving_; }
    std::shared_ptr<Controller> getStandby() override { return standby_; }
    std::shared_ptr<Controller> getValidation() override { return validation_; }

    std::shared_ptr<Controller> getByName(const std::string &name) override
    {
        if (name == "A" || name == "B" || name == "C")
        {
            auto g1 = groupIds_.empty() ? "G1" : groupIds_[0];
            auto list = groups_[g1];
            if (name == "A" && !list.empty())
                return controllers_[list[0]];
            if (name == "B" && list.size() > 1)
                return controllers_[list[1]];
            if (name == "C" && list.size() > 2)
                return controllers_[list[2]];
        }
        auto it = controllers_.find(name);
        return it == controllers_.end() ? nullptr : it->second;
    }

    std::vector<std::string> listControllerNames() const override
    {
        std::vector<std::string> out;
        for (auto &gid : groupIds_)
        {
            auto it = groups_.find(gid);
            if (it == groups_.end())
                continue;
            for (auto &name : it->second)
                out.push_back(name);
        }
        return out;
    }

    std::vector<std::string> listGroupIds() const override { return groupIds_; }

    std::vector<std::string> listControllersInGroup(const std::string &gid) override
    {
        return groups_.count(gid) ? groups_.at(gid) : std::vector<std::string>{};
    }

    std::vector<json> listMetrics() const override
    {
        std::vector<json> out;
        for (auto &name : listControllerNames())
        {
            auto it = controllers_.find(name);
            if (it != controllers_.end())
                out.push_back(it->second->metrics());
        }
        return out;
    }

    json ingestDocument(const json &doc) override
    {
        json results = json::array();
        for (auto &name : listControllerNames())
        {
            auto ctrl = getByName(name);
            if (!ctrl)
            {
                results.push_back(json{{"ok", false}, {"reason", "controller-not-found"}});
                continue;
            }
            results.push_back(ctrl->runtime()->ingestDocument(doc));
        }
        return results;
    }

    json ingestDocumentToGroup(const std::string &gid, const json &doc) override
    {
        auto list = listControllersInGroup(gid);
        if (list.empty())
            return json{{"ok", false}, {"reason", "group-not-found"}};
        json results = json::array();
        for (auto &name : list)
        {
            auto ctrl = getByName(name);
            if (!ctrl)
            {
                results.push_back(json{{"ok", false}, {"reason", "controller-not-found"}});
                continue;
            }
            results.push_back(ctrl->runtime()->ingestDocument(doc));
        }
        return results;
    }

    json forgetMemes(const json &payload) override
    {
        json results = json::array();
        for (auto &name : listControllerNames())
        {
            auto ctrl = getByName(name);
            if (!ctrl)
            {
                results.push_back(json{{"ok", false}, {"reason", "controller-not-found"}});
                continue;
            }
            results.push_back(ctrl->runtime()->forgetMemes(payload));
        }
        return results;
    }

    json onlineResearch(const json &queryOrTokens, const json &options) override
    {
        return serving_->runtime()->onlineLookup(queryOrTokens, options);
    }

    void hotSwap(const json &snapshot) override
    {
        standby_->applySnapshot(snapshot);
        std::swap(serving_, standby_);
    }

    const Config &config() const override { return config_; }

    json respondByName(const std::string &controllerName, const json &payload) override
    {
        if (inferencePool_)
        {
            return inferencePool_->respond(controllerName, payload);
        }
        auto ctrl = getByName(controllerName);
        if (!ctrl)
            throw std::runtime_error("controller-offline");
        return ctrl->respond(payload);
    }

private:
    Config config_;
    std::unordered_map<std::string, std::shared_ptr<Controller>> controllers_;
    std::unordered_map<std::string, std::vector<std::string>> groups_;
    std::vector<std::string> groupIds_;
    std::shared_ptr<Controller> serving_;
    std::shared_ptr<Controller> standby_;
    std::shared_ptr<Controller> validation_;
    std::unique_ptr<InferenceProcessPool> inferencePool_;
};

InferenceProcessPool::InferenceProcessPool(ControllerPool *pool, int workers, int timeoutMs)
    : pool_(pool), timeoutMs_(std::max(200, timeoutMs))
{
    int count = std::max(1, workers);
    running_ = true;
    workers_.reserve(count);
    for (int i = 0; i < count; i++)
    {
        workers_.emplace_back([this]()
                              { workerLoop(); });
    }
}

InferenceProcessPool::~InferenceProcessPool()
{
    stop();
}

void InferenceProcessPool::stop()
{
    if (!running_)
        return;
    running_ = false;
    cv_.notify_all();
    for (auto &t : workers_)
    {
        if (t.joinable())
            t.join();
    }
    workers_.clear();
}

json InferenceProcessPool::respond(const std::string &controllerName, const json &payload)
{
    if (!running_)
        throw std::runtime_error("inference-pool-stopped");
    auto task = std::make_shared<Task>();
    task->controllerName = controllerName;
    auto packed = std::make_shared<std::vector<uint8_t>>(json::to_msgpack(
        buildShmEnvelope(payload, controllerName, pool_ ? pool_->config().transformerBackend : std::string(), "request")));
    task->payload = SharedMemorySlice::fromVector(packed);
    task->promise = std::make_shared<std::promise<SharedMemorySlice>>();
    auto fut = task->promise->get_future();
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (queue_.size() >= maxQueue_)
            throw std::runtime_error("inference-queue-busy");
        queue_.push(task);
    }
    cv_.notify_one();
    if (fut.wait_for(std::chrono::milliseconds(timeoutMs_)) == std::future_status::timeout)
    {
        throw std::runtime_error("inference-timeout");
    }
    auto out = fut.get();
    if (!out.owner)
        return json::object();
    try
    {
        json decoded = json::from_msgpack(*out.owner);
        json payloadDecoded;
        if (unpackShmEnvelope(decoded, payloadDecoded))
            return payloadDecoded;
        return decoded;
    }
    catch (...)
    {
        throw std::runtime_error("inference-invalid-response");
    }
}

void InferenceProcessPool::workerLoop()
{
    while (running_)
    {
        std::shared_ptr<Task> task;
        {
            std::unique_lock<std::mutex> lock(mu_);
            cv_.wait(lock, [&]()
                     { return !running_ || !queue_.empty(); });
            if (!running_ && queue_.empty())
                return;
            task = queue_.front();
            queue_.pop();
        }
        try
        {
            if (!pool_)
                throw std::runtime_error("pool-null");
            auto ctrl = pool_->getByName(task->controllerName);
            if (!ctrl)
                throw std::runtime_error("controller-offline");
            json payload = json::object();
            if (!task->payload.empty() && task->payload.owner)
            {
                auto raw = json::from_msgpack(*task->payload.owner);
                if (!unpackShmEnvelope(raw, payload))
                    payload = raw;
            }
            auto out = ctrl->respond(payload);
            auto packedOut = std::make_shared<std::vector<uint8_t>>(json::to_msgpack(
                buildShmEnvelope(out, task->controllerName, pool_ ? pool_->config().transformerBackend : std::string(), "response")));
            task->promise->set_value(SharedMemorySlice::fromVector(packedOut));
        }
        catch (const std::exception &e)
        {
            try
            {
                task->promise->set_exception(std::make_exception_ptr(std::runtime_error(e.what())));
            }
            catch (...)
            {
            }
        }
    }
}

class GroupProcessPool : public ControllerPoolBase
{
public:
    GroupProcessPool(std::shared_ptr<KeyValueStore> kvmStore,
                     std::shared_ptr<KeyValueStore> memeStore,
                     std::shared_ptr<KeyValueStore> sessionStore,
                     const Config &config)
        : config_(config), kvmStore_(std::move(kvmStore)), memeStore_(std::move(memeStore)), sessionStore_(std::move(sessionStore))
    {
        for (int i = 0; i < config_.groupCount; i++)
        {
            std::string gid = "G" + std::to_string(i + 1);
            groupIds_.push_back(gid);
            std::vector<std::string> names;
            for (int j = 0; j < config_.groupSize; j++)
            {
                std::string name = gid + "_AI" + std::to_string(j + 1);
                names.push_back(name);
                controllerToGroup_[name] = gid;
                controllers_[name] = std::make_shared<ProxyController>(this, name);
            }
            groups_[gid] = names;
        }
        auto g1 = groupIds_.empty() ? "G1" : groupIds_[0];
        auto list = groups_[g1];
        servingName_ = list.empty() ? "" : list[0];
        standbyName_ = list.size() > 1 ? list[1] : servingName_;
        validationName_ = list.size() > 2 ? list[2] : servingName_;
        spawnWorkers();
    }

    ~GroupProcessPool() override
    {
        shutdownWorkers();
    }

    std::shared_ptr<Controller> getActive() override { return getByName(servingName_); }
    std::shared_ptr<Controller> getStandby() override { return getByName(standbyName_); }
    std::shared_ptr<Controller> getValidation() override { return getByName(validationName_); }

    std::shared_ptr<Controller> getByName(const std::string &name) override
    {
        auto it = controllers_.find(name);
        return it == controllers_.end() ? nullptr : it->second;
    }

    std::vector<std::string> listControllerNames() const override
    {
        std::vector<std::string> out;
        for (auto &gid : groupIds_)
        {
            auto it = groups_.find(gid);
            if (it == groups_.end())
                continue;
            for (auto &name : it->second)
                out.push_back(name);
        }
        return out;
    }

    std::vector<std::string> listGroupIds() const override { return groupIds_; }

    std::vector<std::string> listControllersInGroup(const std::string &gid) override
    {
        return groups_.count(gid) ? groups_.at(gid) : std::vector<std::string>{};
    }

    std::vector<json> listMetrics() const override
    {
        std::vector<json> out;
        for (auto &gid : groupIds_)
        {
            try
            {
                auto result = rpc(gid, json{{"cmd", "listMetrics"}});
                if (result.is_array())
                {
                    for (auto &item : result)
                        out.push_back(item);
                }
            }
            catch (...)
            {
                // ignore failed groups
            }
        }
        return out;
    }

    json ingestDocument(const json &doc) override
    {
        json results = json::array();
        for (auto &gid : groupIds_)
        {
            auto part = rpc(gid, json{{"cmd", "ingestDocument"}, {"doc", doc}});
            if (part.is_array())
            {
                for (auto &item : part)
                    results.push_back(item);
            }
            else
            {
                results.push_back(part);
            }
        }
        return results;
    }

    json ingestDocumentToGroup(const std::string &gid, const json &doc) override
    {
        if (!groups_.count(gid))
            return json{{"ok", false}, {"reason", "group-not-found"}};
        return rpc(gid, json{{"cmd", "ingestDocumentToGroup"}, {"doc", doc}});
    }

    json forgetMemes(const json &payload) override
    {
        json results = json::array();
        for (auto &gid : groupIds_)
        {
            auto part = rpc(gid, json{{"cmd", "forgetMemes"}, {"criteria", payload}});
            if (part.is_array())
            {
                for (auto &item : part)
                    results.push_back(item);
            }
            else
            {
                results.push_back(part);
            }
        }
        return results;
    }

    json onlineResearch(const json &queryOrTokens, const json &options) override
    {
        auto ctrl = getActive();
        if (!ctrl)
            throw std::runtime_error("no-active-controller");
        return rpc(controllerToGroup_[ctrl->name()], json{{"cmd", "runtimeCall"}, {"controllerName", ctrl->name()}, {"method", "onlineLookup"}, {"args", json::array({queryOrTokens, options})}});
    }

    void hotSwap(const json &snapshot) override
    {
        for (auto &gid : groupIds_)
        {
            rpc(gid, json{{"cmd", "applySnapshotAll"}, {"snapshot", snapshot}});
        }
        std::swap(servingName_, standbyName_);
    }

    void applySnapshotAll(const json &snapshot)
    {
        for (auto &gid : groupIds_)
        {
            rpc(gid, json{{"cmd", "applySnapshotAll"}, {"snapshot", snapshot}});
        }
    }

    const Config &config() const override { return config_; }

    json respondByName(const std::string &controllerName, const json &payload) override
    {
        auto gid = controllerToGroup_.count(controllerName) ? controllerToGroup_[controllerName] : std::string();
        if (gid.empty())
            throw std::runtime_error("group-not-found");
        return rpc(gid, json{{"cmd", "respond"}, {"controllerName", controllerName}, {"payload", payload}});
    }

    json rpc(const std::string &groupId, const json &payload) const
    {
        auto it = workers_.find(groupId);
        if (it == workers_.end())
            throw std::runtime_error("group-worker-unavailable:" + groupId);
        auto &worker = *it->second;
        std::lock_guard<std::mutex> lock(worker.mu);
        json msg = payload;
        msg["id"] = std::to_string(worker.seq++);
        msg["shmProtocolVersion"] = kSharedMemoryProtocolVersion;
        msg["transport"] = "socket-msgpack-shm2";
        msg["backendHint"] = config_.transformerBackend;
        std::vector<uint8_t> requestBytes = json::to_msgpack(msg);
        auto attempt = [&](bool reconnect) -> std::optional<json>
        {
            if (reconnect)
            {
                closeSocket(worker.sock);
                worker.sock = connectWorker(worker.port);
            }
            if (worker.sock == kInvalidSocket)
                worker.sock = connectWorker(worker.port);
            if (worker.sock == kInvalidSocket || !sendPacket(worker.sock, requestBytes))
                return std::nullopt;

            std::vector<uint8_t> replyBytes;
            if (!recvPacket(worker.sock, replyBytes, config_.groupProcTimeoutMs))
                return std::nullopt;

            json reply;
            try
            {
                reply = json::from_msgpack(replyBytes);
            }
            catch (...)
            {
                return std::nullopt;
            }
            if (reply.contains("id") && msg.contains("id") && reply["id"].is_string() && msg["id"].is_string())
            {
                if (reply["id"].get<std::string>() != msg["id"].get<std::string>())
                    return std::nullopt;
            }
            if (reply.contains("shmProtocolVersion") && reply["shmProtocolVersion"].is_number_integer())
            {
                int proto = reply["shmProtocolVersion"].get<int>();
                if (proto <= 0 || proto > kSharedMemoryProtocolVersion)
                    return std::nullopt;
            }
            return reply;
        };

        auto first = attempt(false);
        auto reply = first ? first : attempt(true);
        if (!reply)
        {
            std::string cmd = payload.value("cmd", "");
            throw std::runtime_error("group-worker-timeout:" + groupId + ":" + cmd);
        }
        if ((*reply).value("ok", false))
            return (*reply).value("result", json::object());
        std::string err = (*reply).value("error", "");
        if (err.empty())
            err = "group-worker-error:" + groupId;
        throw std::runtime_error(err);
    }

    json runtimeCall(const std::string &controllerName, const std::string &method, const json &args) const
    {
        auto gid = controllerToGroup_.count(controllerName) ? controllerToGroup_.at(controllerName) : std::string();
        if (gid.empty())
            throw std::runtime_error("group-not-found");
        return rpc(gid, json{{"cmd", "runtimeCall"}, {"controllerName", controllerName}, {"method", method}, {"args", args}});
    }

    json memebarrierStart(double threshold) const
    {
        auto gid = controllerToGroup_.count(servingName_) ? controllerToGroup_.at(servingName_) : std::string();
        if (gid.empty())
            throw std::runtime_error("group-not-found");
        return rpc(gid, json{{"cmd", "memebarrierStart"}, {"maliciousThreshold", threshold}});
    }

    json memebarrierStop() const
    {
        auto gid = controllerToGroup_.count(servingName_) ? controllerToGroup_.at(servingName_) : std::string();
        if (gid.empty())
            throw std::runtime_error("group-not-found");
        return rpc(gid, json{{"cmd", "memebarrierStop"}});
    }

    json memebarrierStats() const
    {
        auto gid = controllerToGroup_.count(servingName_) ? controllerToGroup_.at(servingName_) : std::string();
        if (gid.empty())
            throw std::runtime_error("group-not-found");
        return rpc(gid, json{{"cmd", "memebarrierStats"}});
    }

    json memebarrierSetThreshold(double threshold) const
    {
        auto gid = controllerToGroup_.count(servingName_) ? controllerToGroup_.at(servingName_) : std::string();
        if (gid.empty())
            throw std::runtime_error("group-not-found");
        return rpc(gid, json{{"cmd", "memebarrierSetThreshold"}, {"maliciousThreshold", threshold}});
    }

    std::string groupForController(const std::string &name) const
    {
        auto it = controllerToGroup_.find(name);
        return it == controllerToGroup_.end() ? std::string() : it->second;
    }

private:
    struct WorkerInfo
    {
        int port{0};
        mutable SocketHandle sock{kInvalidSocket};
        mutable std::mutex mu;
        mutable uint64_t seq{1};
#ifdef _WIN32
        PROCESS_INFORMATION proc{0};
#else
        pid_t pid{-1};
#endif
    };

    Config config_;
    std::shared_ptr<KeyValueStore> kvmStore_;
    std::shared_ptr<KeyValueStore> memeStore_;
    std::shared_ptr<KeyValueStore> sessionStore_;
    std::vector<std::string> groupIds_;
    std::unordered_map<std::string, std::vector<std::string>> groups_;
    std::unordered_map<std::string, std::string> controllerToGroup_;
    std::unordered_map<std::string, std::shared_ptr<Controller>> controllers_;
    mutable std::unordered_map<std::string, std::shared_ptr<WorkerInfo>> workers_;
    std::string servingName_;
    std::string standbyName_;
    std::string validationName_;

    void spawnWorkers()
    {
        int basePort = std::max(1, config_.portStudy + 100);
        for (size_t i = 0; i < groupIds_.size(); i++)
        {
            std::string gid = groupIds_[i];
            int port = basePort + (int)i;
            auto info = std::make_shared<WorkerInfo>();
            info->port = port;
            spawnWorkerProcess(gid, port, *info);
            info->sock = connectWorker(port);
            workers_[gid] = info;
        }
    }

    void shutdownWorkers()
    {
        for (auto &kv : workers_)
        {
            if (!kv.second)
                continue;
            auto &worker = *kv.second;
            closeSocket(worker.sock);
#ifdef _WIN32
            if (worker.proc.hProcess)
            {
                TerminateProcess(worker.proc.hProcess, 0);
                CloseHandle(worker.proc.hProcess);
                CloseHandle(worker.proc.hThread);
            }
#else
            if (worker.pid > 0)
            {
                kill(worker.pid, SIGTERM);
            }
#endif
        }
    }

    void spawnWorkerProcess(const std::string &groupId, int port, WorkerInfo &info) const
    {
        std::string portStr = std::to_string(port);
        std::string groupSize = std::to_string(config_.groupSize);
#ifdef _WIN32
        STARTUPINFOA si{};
        PROCESS_INFORMATION pi{};
        si.cb = sizeof(si);
        std::ostringstream cmd;
        cmd << '"' << g_selfPath << '"' << " --group-worker=1";
        std::string cmdLine = cmd.str();

        std::string envBlock;
        auto addEnv = [&](const std::string &k, const std::string &v)
        {
            envBlock += k + "=" + v + "\0";
        };
        addEnv("AI_GROUP_ID", groupId);
        addEnv("AI_GROUP_SIZE", groupSize);
        addEnv("AI_GROUP_COUNT", std::to_string(config_.groupCount));
        addEnv("AI_LMDB_ROOT", config_.lmdbRoot.string());
        addEnv("AI_LMDB_MAP_BYTES", std::to_string(config_.lmdbMapSizeBytes));
        addEnv("AI_BASE_DIR", config_.baseDir.string());
        addEnv("AI_GROUP_WORKER_PORT", portStr);
        envBlock.push_back('\0');

        BOOL ok = CreateProcessA(nullptr, cmdLine.data(), nullptr, nullptr, FALSE, 0, (LPVOID)envBlock.c_str(), nullptr, &si, &pi);
        if (!ok)
        {
            throw std::runtime_error("failed-to-spawn-group-worker");
        }
        info.proc = pi;
        CloseHandle(pi.hThread);
#else
        pid_t pid = fork();
        if (pid == 0)
        {
            setenv("AI_GROUP_ID", groupId.c_str(), 1);
            setenv("AI_GROUP_SIZE", groupSize.c_str(), 1);
            setenv("AI_GROUP_COUNT", std::to_string(config_.groupCount).c_str(), 1);
            setenv("AI_LMDB_ROOT", config_.lmdbRoot.string().c_str(), 1);
            setenv("AI_LMDB_MAP_BYTES", std::to_string(config_.lmdbMapSizeBytes).c_str(), 1);
            setenv("AI_BASE_DIR", config_.baseDir.string().c_str(), 1);
            setenv("AI_GROUP_WORKER_PORT", portStr.c_str(), 1);
            execl(g_selfPath.c_str(), g_selfPath.c_str(), "--group-worker=1", nullptr);
            _exit(1);
        }
        info.pid = pid;
#endif
    }

    SocketHandle connectWorker(int port) const
    {
#ifdef _WIN32
        static std::atomic<bool> wsaReady{false};
        if (!wsaReady)
        {
            WSADATA wsaData{};
            WSAStartup(MAKEWORD(2, 2), &wsaData);
            wsaReady = true;
        }
#endif
        SocketHandle sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock == kInvalidSocket)
            return kInvalidSocket;
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons((uint16_t)port);
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        for (int i = 0; i < 50; i++)
        {
            if (connect(sock, (sockaddr *)&addr, sizeof(addr)) == 0)
            {
                return sock;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        closeSocket(sock);
        return kInvalidSocket;
    }
};

json ProxyRuntime::processInput(const json &payload)
{
    return pool_->respondByName(controllerName_, payload);
}

json ProxyRuntime::cloneParams() const
{
    return pool_->rpc(pool_->groupForController(controllerName_), json{{"cmd", "cloneParams"}, {"controllerName", controllerName_}});
}

void ProxyRuntime::setParams(const json &patch)
{
    pool_->rpc(pool_->groupForController(controllerName_), json{{"cmd", "applyParams"}, {"controllerName", controllerName_}, {"params", patch}});
}

json ProxyRuntime::toSnapshot()
{
    return pool_->rpc(pool_->groupForController(controllerName_), json{{"cmd", "snapshot"}, {"controllerName", controllerName_}});
}

void ProxyRuntime::fromSnapshot(const json &snapshot)
{
    pool_->rpc(pool_->groupForController(controllerName_), json{{"cmd", "applySnapshot"}, {"controllerName", controllerName_}, {"snapshot", snapshot}});
}

json ProxyRuntime::ingestDocument(const json &doc)
{
    return pool_->rpc(pool_->groupForController(controllerName_), json{{"cmd", "ingestDocumentTo"}, {"controllerName", controllerName_}, {"doc", doc}});
}

json ProxyRuntime::forgetMemes(const json &payload)
{
    return pool_->forgetMemes(payload);
}

json ProxyRuntime::onlineLookup(const std::string &query, const json &options)
{
    return pool_->runtimeCall(controllerName_, "onlineLookup", json::array({query, options}));
}

json ProxyRuntime::onlineLookup(const json &input, const json &options)
{
    return pool_->runtimeCall(controllerName_, "onlineLookup", json::array({input, options}));
}

json ProxyRuntime::getSearchConfig() const
{
    return pool_->runtimeCall(controllerName_, "getSearchConfig", json::array());
}

json ProxyRuntime::setSearchConfig(const json &patch)
{
    return pool_->runtimeCall(controllerName_, "setSearchConfig", json::array({patch}));
}

std::vector<RobotsDocument> ProxyRuntime::collectRobotsDocuments(const json &options) const
{
    auto out = pool_->runtimeCall(controllerName_, "collectRobotsDocuments", json::array({options}));
    std::vector<RobotsDocument> docs;
    if (out.is_array())
    {
        for (auto &item : out)
        {
            RobotsDocument d;
            d.id = item.value("id", "");
            d.text = item.value("text", "");
            d.tokens = item.value("tokens", std::vector<std::string>{});
            d.source = item.value("source", "");
            d.file = item.value("file", "");
            docs.push_back(std::move(d));
        }
    }
    return docs;
}

std::vector<std::string> ProxyRuntime::listRobotsFiles() const
{
    auto out = pool_->runtimeCall(controllerName_, "listRobotsFiles", json::array());
    std::vector<std::string> files;
    if (out.is_array())
    {
        for (auto &v : out)
            if (v.is_string())
                files.push_back(v.get<std::string>());
    }
    return files;
}

fs::path ProxyRuntime::exportGraphToFile(const json &options)
{
    auto out = pool_->runtimeCall(controllerName_, "exportGraphToFile", json::array({options}));
    if (out.is_string())
        return fs::path(out.get<std::string>());
    if (out.is_object() && out.contains("path"))
        return fs::path(out["path"].get<std::string>());
    return {};
}

json ProxyRuntime::learnFromDialog(const json &payload, const json &result)
{
    return pool_->runtimeCall(controllerName_, "learnFromDialog", json::array({json{{"payload", payload}, {"result", result}}}));
}

std::unordered_map<std::string, double> ProxyRuntime::mapWordsToMemes(const std::vector<std::string> &words)
{
    auto out = pool_->runtimeCall(controllerName_, "mapWordsToMemes", json::array({words}));
    std::unordered_map<std::string, double> res;
    if (out.is_object())
    {
        for (auto it = out.begin(); it != out.end(); ++it)
        {
            if (it.value().is_number())
                res[it.key()] = it.value().get<double>();
        }
    }
    return res;
}

std::vector<std::string> ProxyRuntime::getMemeWords(const std::string &memeId)
{
    auto out = pool_->runtimeCall(controllerName_, "getMemeWords", json::array({memeId}));
    std::vector<std::string> res;
    if (out.is_array())
    {
        for (auto &v : out)
            if (v.is_string())
                res.push_back(v.get<std::string>());
    }
    return res;
}

void ProxyRuntime::graphLink(const std::string &fromId, const std::string &toId, double weight, int direction)
{
    pool_->runtimeCall(controllerName_, "graphLink", json::array({fromId, toId, weight, direction}));
}

const Config &ProxyRuntime::config() const
{
    return pool_->config();
}

json ProxyController::respond(const json &payload)
{
    return pool_->respondByName(name_, payload);
}

json ProxyController::metrics() const
{
    return pool_->rpc(pool_->groupForController(name_), json{{"cmd", "metrics"}, {"controllerName", name_}});
}

void ProxyController::applyParams(const json &p)
{
    pool_->rpc(pool_->groupForController(name_), json{{"cmd", "applyParams"}, {"controllerName", name_}, {"params", p}});
}

json ProxyController::snapshot() const
{
    return pool_->rpc(pool_->groupForController(name_), json{{"cmd", "snapshot"}, {"controllerName", name_}});
}

void ProxyController::applySnapshot(const json &snap)
{
    pool_->rpc(pool_->groupForController(name_), json{{"cmd", "applySnapshot"}, {"controllerName", name_}, {"snapshot", snap}});
}

// ------------------ Spark Arrays ------------------

// PersonaForestAverager 负责多候选回复的森林式聚合决策。
// 调用方式：接收层级候选后返回最终回复与解释信息。
// 实现思路：子空间随机采样 + 投票评分融合。
// 注意事项：特征权重与样本率会影响稳定性与多样性。
// 注意事项：候选过少时会自动降级为直接返回。
// 注意事项：该模块依赖输入候选质量。
class PersonaForestAverager : public IPersonaForestAverager
{
public:
    explicit PersonaForestAverager(const json &options = json::object())
    {
        enabled_ = options.value("enabled", true);
        trees_ = std::max(8, options.value("trees", 32));
        featureSubspace_ = std::max(2, options.value("featureSubspace", 4));
        sampleRate_ = clamp01(options.value("sampleRate", 0.85));
        personaMomentum_ = clamp01(options.value("personaMomentum", 0.92));
        targetReplyLen_ = std::max(20, options.value("targetReplyLen", 220));
        maxHistory_ = std::max(10, options.value("maxHistory", 60));
    }

    int maxHistory() const override { return maxHistory_; }

    std::optional<json> pick(const json &payload, const json &layerResults, const std::vector<float> &requestEmbedding, const std::vector<json> &history) override
    {
        if (!enabled_)
            return std::nullopt;
        int dim = requestEmbedding.empty() ? 64 : (int)requestEmbedding.size();
        std::string inputText = payload.value("text", "");
        auto inputTokens = tokenize(inputText);

        std::vector<Candidate> candidates;
        if (layerResults.is_array())
        {
            for (auto &layer : layerResults)
            {
                auto controllers = layer.value("controllers", json::array());
                for (auto &controller : controllers)
                {
                    auto base = controller.value("base", json::object());
                    if (!base.contains("reply"))
                        continue;
                    std::string reply = base.value("reply", "");
                    if (reply.empty())
                        continue;
                    Candidate c;
                    c.layer = layer.value("layer", "");
                    c.controller = controller.value("controller", "");
                    c.reply = reply;
                    c.latency = base.value("latency", 0);
                    c.sessionId = base.value("sessionId", "");
                    c.emb = textToMiniEmbedding(reply, dim);
                    if (controller.contains("affinity") && controller["affinity"].is_number())
                    {
                        c.affinity = clamp11(controller["affinity"].get<double>());
                    }
                    else
                    {
                        c.affinity = clamp11(cosineSim(requestEmbedding, c.emb));
                    }
                    c.controllerRef = controller;
                    candidates.push_back(std::move(c));
                }
            }
        }

        if (candidates.empty())
            return std::nullopt;
        if (candidates.size() == 1)
        {
            updatePersonas(candidates[0]);
            return json{{"layer", candidates[0].layer}, {"controller", candidates[0].controller}, {"affinity", candidates[0].affinity}, {"reply", candidates[0].reply}, {"latency", candidates[0].latency}, {"sessionId", candidates[0].sessionId}, {"score", 0}, {"method", "forest-single"}};
        }

        auto centroid = centroidEmbedding(candidates, dim);
        json lastAgg = history.empty() ? json::object() : history.back().value("aggregate", json::object());
        std::vector<float> lastReplyEmb;
        if (lastAgg.contains("reply"))
            lastReplyEmb = textToMiniEmbedding(lastAgg.value("reply", ""), dim);
        int lastReplyLen = lastAgg.contains("reply") ? (int)lastAgg.value("reply", "").size() : 0;
        int targetLen = (lastReplyLen > 20) ? lastReplyLen : targetReplyLen_;

        for (auto &c : candidates)
        {
            auto replyTokens = tokenize(c.reply);
            double overlap = tokenOverlapRatio(inputTokens, replyTokens);
            double consensus = centroid.empty() ? 0.0 : clamp11(cosineSim(c.emb, centroid));
            double persona = globalPersona_.empty() ? 0.0 : clamp11(cosineSim(c.emb, globalPersona_));
            auto it = controllerPersona_.find(c.controller);
            double controllerPersona = (it == controllerPersona_.end() || it->second.empty()) ? 0.0 : clamp11(cosineSim(c.emb, it->second));
            double novelty = lastReplyEmb.empty() ? 0.5 : clamp01(1 - ((clamp11(cosineSim(c.emb, lastReplyEmb)) + 1) / 2));
            double lenPenalty = -clamp01(std::abs((int)c.reply.size() - targetLen) / (double)std::max(40, targetLen));
            double stability = variantStability(c.controllerRef);
            double latencyNorm = c.latency > 0 ? clamp01(1 - (c.latency / 2000.0)) : 0.5;
            c.features = json{{"affinity", clamp01((c.affinity + 1) / 2)}, {"overlap", overlap}, {"consensus", clamp01((consensus + 1) / 2)}, {"persona", clamp01((persona + 1) / 2)}, {"controllerPersona", clamp01((controllerPersona + 1) / 2)}, {"novelty", novelty}, {"lenPenalty", clamp01(1 + lenPenalty)}, {"stability", stability}, {"latency", latencyNorm}};
        }

        auto featureNames = json::array({"affinity", "overlap", "consensus", "persona", "controllerPersona", "novelty", "lenPenalty", "stability", "latency"});
        std::string seedText = payload.value("sessionId", "") + "|" + std::to_string(hashStrSimple(inputText)) + "|" + std::to_string(candidates.size());
        Rng32 rng(hashStrSimple(seedText));

        std::unordered_map<int, double> scores;
        std::unordered_map<int, int> votes;
        for (size_t i = 0; i < candidates.size(); i++)
        {
            scores[(int)i] = 0;
            votes[(int)i] = 0;
        }

        auto pickSubspace = [&]()
        {
            std::unordered_set<int> chosen;
            int maxPick = std::min(featureSubspace_, (int)featureNames.size());
            while ((int)chosen.size() < maxPick)
            {
                int idx = (int)std::floor(rng.next() * featureNames.size());
                chosen.insert(idx);
            }
            std::vector<int> out(chosen.begin(), chosen.end());
            return out;
        };

        for (int t = 0; t < trees_; t++)
        {
            auto subspace = pickSubspace();
            std::unordered_map<std::string, double> weights;
            for (auto idx : subspace)
            {
                std::string f = featureNames[idx].get<std::string>();
                double w = (rng.next() * 2 - 1);
                if (f == "affinity" || f == "consensus" || f == "persona")
                    w *= 1.4;
                if (f == "controllerPersona" || f == "stability")
                    w *= 1.1;
                if (f == "novelty")
                    w *= 0.6;
                weights[f] = w;
            }

            std::vector<int> bag;
            for (size_t i = 0; i < candidates.size(); i++)
            {
                if (rng.next() <= sampleRate_)
                    bag.push_back((int)i);
            }
            if (bag.size() < 2)
            {
                bag.clear();
                for (size_t i = 0; i < candidates.size(); i++)
                    bag.push_back((int)i);
            }

            int bestIdx = -1;
            double bestScore = -1e9;
            for (auto idx : bag)
            {
                double s = 0.0;
                for (auto &kv : weights)
                {
                    double v = candidates[idx].features.value(kv.first, 0.0);
                    s += kv.second * v;
                }
                if (s > bestScore)
                {
                    bestScore = s;
                    bestIdx = idx;
                }
            }
            if (bestIdx >= 0)
                votes[bestIdx] += 1;
            for (size_t i = 0; i < candidates.size(); i++)
            {
                double s = 0.0;
                for (auto &kv : weights)
                {
                    double v = candidates[i].features.value(kv.first, 0.0);
                    s += kv.second * v;
                }
                scores[(int)i] += s;
            }
        }

        int winner = -1;
        for (size_t i = 0; i < candidates.size(); i++)
        {
            if (winner == -1)
            {
                winner = (int)i;
                continue;
            }
            int vi = votes[(int)i];
            int vw = votes[winner];
            double si = scores[(int)i];
            double sw = scores[winner];
            double ci = candidates[i].features.value("consensus", 0.0);
            double cw = candidates[winner].features.value("consensus", 0.0);
            double ai = candidates[i].features.value("affinity", 0.0);
            double aw = candidates[winner].features.value("affinity", 0.0);
            if (vi > vw || (vi == vw && (si > sw || (si == sw && (ci > cw || (ci == cw && ai > aw))))))
            {
                winner = (int)i;
            }
        }

        if (winner < 0)
            return std::nullopt;
        updatePersonas(candidates[winner]);
        return json{{"layer", candidates[winner].layer}, {"controller", candidates[winner].controller}, {"affinity", candidates[winner].affinity}, {"reply", candidates[winner].reply}, {"latency", candidates[winner].latency}, {"sessionId", candidates[winner].sessionId}, {"score", scores[winner] / std::max(1, trees_)}, {"votes", votes[winner]}, {"method", "persona-forest"}};
    }

private:
    struct Candidate
    {
        std::string layer;
        std::string controller;
        std::string reply;
        int latency{0};
        std::string sessionId;
        double affinity{0};
        std::vector<float> emb;
        json controllerRef;
        json features;
    };
    bool enabled_{true};
    int trees_{32};
    int featureSubspace_{4};
    double sampleRate_{0.85};
    double personaMomentum_{0.92};
    int targetReplyLen_{220};
    int maxHistory_{60};
    std::vector<float> globalPersona_;
    std::unordered_map<std::string, std::vector<float>> controllerPersona_;

    static std::vector<float> centroidEmbedding(const std::vector<Candidate> &cands, int dim)
    {
        if (cands.empty())
            return {};
        std::vector<float> out(dim, 0.0f);
        int count = 0;
        for (auto &c : cands)
        {
            if ((int)c.emb.size() != dim)
                continue;
            for (int i = 0; i < dim; i++)
                out[i] += c.emb[i];
            count++;
        }
        if (!count)
            return {};
        for (int i = 0; i < dim; i++)
            out[i] /= count;
        return out;
    }

    static double tokenOverlapRatio(const std::vector<std::string> &a, const std::vector<std::string> &b)
    {
        if (a.empty() || b.empty())
            return 0.0;
        std::unordered_set<std::string> setB(b.begin(), b.end());
        double hit = 0.0;
        for (auto &t : a)
            if (setB.count(t))
                hit += 1.0;
        return hit / std::max(1.0, (double)a.size());
    }

    double variantStability(const json &controllerRef) const
    {
        auto variants = controllerRef.value("variants", json::array());
        if (!variants.is_array() || variants.empty())
            return 0.5;
        double sum = 0.0;
        int n = 0;
        for (auto &v : variants)
        {
            if (!v.is_object() || !v.contains("affinity"))
                continue;
            if (!v["affinity"].is_number())
                continue;
            double a = v["affinity"].get<double>();
            if (!std::isfinite(a))
                continue;
            sum += clamp11(a);
            n++;
        }
        if (!n)
            return 0.5;
        return clamp01((sum / n + 1) / 2);
    }

    void updatePersonas(const Candidate &winner)
    {
        if (winner.emb.empty())
            return;
        auto blend = [&](const std::vector<float> &prev, const std::vector<float> &next)
        {
            if (next.empty())
                return prev;
            if (prev.empty())
                return next;
            std::vector<float> out(prev.size(), 0.0f);
            for (size_t i = 0; i < prev.size(); i++)
            {
                double v = personaMomentum_ * prev[i] + (1 - personaMomentum_) * next[i];
                out[i] = std::isfinite(v) ? (float)v : 0.0f;
            }
            return out;
        };
        globalPersona_ = blend(globalPersona_, winner.emb);
        auto it = controllerPersona_.find(winner.controller);
        if (it == controllerPersona_.end())
            controllerPersona_[winner.controller] = blend({}, winner.emb);
        else
            it->second = blend(it->second, winner.emb);

        if (controllerPersona_.size() > 64)
        {
            std::vector<std::string> keys;
            keys.reserve(controllerPersona_.size());
            for (auto &kv : controllerPersona_)
                keys.push_back(kv.first);
            while (controllerPersona_.size() > 64)
            {
                size_t idx = (size_t)(std::rand() % keys.size());
                controllerPersona_.erase(keys[idx]);
            }
        }
    }
};

class SparkArray : public ISparkArray
{
public:
    SparkArray(std::shared_ptr<ControllerPoolBase> pool, std::shared_ptr<ShardManager> shards, const json &options = json::object())
        : pool_(std::move(pool)), shards_(std::move(shards))
    {
        auto personaOptions = options.value("personaForest", json::object());
        auto personaFactory = module_mount::ModuleRegistry::instance().personaForestAveragerFactory();
        if (personaFactory)
            personaForest_ = personaFactory(personaOptions);
        else
            personaForest_ = std::make_shared<PersonaForestAverager>(personaOptions);

        groupId_ = options.value("groupId", std::string("G1"));
        rfInstances_ = std::max(1, std::min(32, options.value("rfInstances", options.value("numAI", 7))));
        std::vector<std::string> available;
        if (options.contains("groupIds") && options["groupIds"].is_array() && !options["groupIds"].empty())
        {
            for (auto &gid : options["groupIds"])
            {
                if (!gid.is_string())
                    continue;
                auto vec = pool_->listControllersInGroup(gid.get<std::string>());
                available.insert(available.end(), vec.begin(), vec.end());
            }
        }
        if (available.empty())
            available = pool_->listControllersInGroup(groupId_);
        auto wantedRaw = options.contains("numAI") ? options["numAI"].get<int>() : (options.contains("groupSize") ? options["groupSize"].get<int>() : 7);
        int wanted = std::max(1, wantedRaw);
        int numAI = std::max(1, std::min((int)available.size() ? (int)available.size() : wanted, wanted));
        layers_ = json::array();
        for (int i = 0; i < numAI; i++)
        {
            if (i >= (int)available.size())
                break;
            layers_.push_back(json{{"name", groupId_ + ":a" + std::to_string(i + 1)}, {"controllers", json::array({available[i]})}, {"strategy", "max"}});
        }
        if (options.contains("layers") && options["layers"].is_array() && !options["layers"].empty())
        {
            layers_ = json::array();
            for (auto &layer : options["layers"])
            {
                json next = layer;
                if (!next.contains("strategy"))
                    next["strategy"] = "max";
                layers_.push_back(next);
            }
        }
        defaultBudget_ = normalizeBudget(options.value("budget", json()));
    }

    json getLayers() const override { return layers_; }

    void updateLayers(const json &layers) override
    {
        if (!layers.is_array() || layers.empty())
            throw std::runtime_error("layers must be a non-empty array");
        for (auto &layer : layers)
        {
            if (!layer.contains("name") || !layer.contains("controllers") || !layer["controllers"].is_array())
            {
                throw std::runtime_error("invalid layer specification");
            }
        }
        layers_ = json::array();
        for (auto &layer : layers)
        {
            json next = layer;
            if (!next.contains("strategy"))
                next["strategy"] = "max";
            layers_.push_back(next);
        }
    }

    json dispatch(const json &payload, const json &options = json::object()) override
    {
        std::string text = payload.value("text", "");
        auto requestEmbedding = textToMiniEmbedding(text, 64);
        auto variants = buildVariants(text, options.value("perturbations", 0));
        json layers = options.value("multiLayer", true) ? layers_ : json::array({layers_.empty() ? json::object() : layers_[0]});
        if (options.contains("numAI") && options["numAI"].is_number())
        {
            int cap = std::max(1, options["numAI"].get<int>());
            json trimmed = json::array();
            for (int i = 0; i < (int)layers.size() && i < cap; i++)
                trimmed.push_back(layers[i]);
            layers = trimmed;
        }
        auto budget = mergeBudgets(defaultBudget_, normalizeBudget(options.value("budget", payload.value("budget", json()))));

        json layerResults = json::array();

        auto callRespond = [&](const std::string &controllerName, const json &nextPayload)
        {
            return pool_->respondByName(controllerName, nextPayload);
        };

        for (auto &layer : layers)
        {
            json controllers = json::array();
            for (auto &controllerSpec : layer.value("controllers", json::array()))
            {
                std::string controllerName = controllerSpec.is_string() ? controllerSpec.get<std::string>() : controllerSpec.value("name", controllerSpec.value("controller", ""));
                if (controllerName.empty())
                {
                    controllers.push_back(json{{"controller", ""}, {"error", "invalid-controller"}});
                    continue;
                }
                int weight = 1;
                if (controllerSpec.is_object())
                {
                    auto w = controllerSpec.value("weight", controllerSpec.value("w", 1));
                    weight = std::max(1, std::min(8, w));
                }
                std::string weightedText = weight <= 1 ? text : std::string();
                if (weight > 1)
                {
                    for (int i = 0; i < weight; i++)
                    {
                        if (i)
                            weightedText += " ";
                        weightedText += text;
                    }
                }

                json nextPayload = payload;
                nextPayload["text"] = weightedText;
                if (budget)
                    nextPayload["budget"] = *budget;

                json baseResult;
                try
                {
                    baseResult = callRespond(controllerName, nextPayload);
                }
                catch (const std::exception &e)
                {
                    shards_->recordFailure(controllerName);
                    controllers.push_back(json{{"controller", controllerName}, {"error", e.what()}});
                    continue;
                }
                double affinity = cosineSim(requestEmbedding, textToMiniEmbedding(baseResult.value("reply", ""), 64));
                shards_->record(controllerName, requestEmbedding, baseResult.value("reply", ""), baseResult.value("latency", 0));
                json variantResults = json::array();
                for (auto &variant : variants)
                {
                    json vrPayload = payload;
                    vrPayload["text"] = variant;
                    if (budget)
                        vrPayload["budget"] = *budget;
                    try
                    {
                        auto vr = callRespond(controllerName, vrPayload);
                        double vAffinity = cosineSim(textToMiniEmbedding(variant, 64), textToMiniEmbedding(vr.value("reply", ""), 64));
                        variantResults.push_back(json{{"text", variant}, {"response", vr}, {"affinity", vAffinity}});
                    }
                    catch (const std::exception &e)
                    {
                        variantResults.push_back(json{{"text", variant}, {"error", e.what()}});
                    }
                }
                controllers.push_back(json{{"controller", controllerName}, {"base", baseResult}, {"affinity", affinity}, {"variants", variantResults}});
            }
            layerResults.push_back(json{{"layer", layer.value("name", "")}, {"controllers", controllers}});
        }

        auto aggregate = aggregateResults(layerResults, payload, requestEmbedding);
        recordHistory(payload, aggregate, layerResults);
        return json{{"aggregate", aggregate}, {"layers", layerResults}};
    }

    json dispatchBig(const json &payload, const json &options) override
    {
        int rounds = std::max(1, std::min(10, options.value("bigRounds", options.value("bigSparkRounds", 1))));
        json history = json::array();
        json current = payload;
        std::string originalText = payload.value("text", "");
        for (int i = 0; i < rounds; i++)
        {
            auto r = dispatch(current, options);
            history.push_back(r);
            std::string reply = r.value("aggregate", json::object()).value("reply", "");
            if (reply.empty())
                break;
            std::string combined = originalText + "\n" + reply;
            if (combined.size() > 4000)
                combined = combined.substr(combined.size() - 4000);
            current["text"] = combined;
        }
        json last = history.empty() ? json{{"aggregate", nullptr}, {"layers", json::array()}} : history.back();
        last["rounds"] = history;
        return last;
    }

    const std::vector<json> &history() const override { return history_; }

private:
    struct RouteMetric
    {
        uint64_t calls{0};
        uint64_t errors{0};
        uint64_t totalMs{0};
        uint64_t maxMs{0};
        uint64_t lastMs{0};
    };

    std::shared_ptr<ControllerPoolBase> pool_;
    std::shared_ptr<ShardManager> shards_;
    std::string groupId_{"G1"};
    int rfInstances_{7};
    json layers_ = json::array();
    std::shared_ptr<IPersonaForestAverager> personaForest_;
    std::optional<json> defaultBudget_;
    std::vector<json> history_;

    json aggregateResults(const json &layerResults, const json &payload, const std::vector<float> &requestEmbedding)
    {
        std::unordered_map<std::string, double> tokenVotes;
        std::vector<double> rfScores;
        rfScores.reserve((size_t)rfInstances_);
        std::unordered_set<std::string> stop{
            "the", "a", "an", "is", "are", "was", "were", "and", "or", "to", "of", "in", "on", "for", "with", "that", "this", "it", "as", "be", "by", "from", "at", "we", "you", "i"};
        auto requestText = payload.value("text", "");
        auto inputTok = tokenize(requestText);
        json transformerCandidates = payload.value("sparkTransformerCandidates", json::array());
        int voteBudget = std::max(1, rfInstances_ + std::min(8, (int)transformerCandidates.size()));
        int used = 0;
        int gnnUsed = 0;
        int transformerUsed = 0;

        auto scoreAndVoteReply = [&](const std::string &reply, double affinity, double latency, double affinityWeight, double latencyWeight, double overlapWeight, double scoreBoost)
        {
            auto replyTok = tokenize(reply);
            std::unordered_set<std::string> setReply(replyTok.begin(), replyTok.end());
            int hit = 0;
            for (auto &w : inputTok)
            {
                if (setReply.count(w))
                    hit++;
            }
            double overlap = inputTok.empty() ? 0.0 : (double)hit / std::max(1.0, (double)inputTok.size());
            double latencyNorm = clamp01(1.0 - std::max(0.0, latency) / 2000.0);
            double score = affinityWeight * clamp01((affinity + 1.0) * 0.5) + latencyWeight * latencyNorm + overlapWeight * overlap;
            score = clamp01(score * std::max(0.1, scoreBoost));
            rfScores.push_back(score);
            if (used < voteBudget)
            {
                used++;
                double weight = std::max(0.05, score);
                for (auto &t : replyTok)
                {
                    if (t.size() < 2 || t.size() > 24)
                        continue;
                    if (stop.count(t))
                        continue;
                    tokenVotes[t] += weight;
                }
            }
        };

        for (auto &layer : layerResults)
        {
            for (auto &controller : layer.value("controllers", json::array()))
            {
                if (!controller.contains("base"))
                    continue;
                auto base = controller["base"];
                if (!base.contains("reply"))
                    continue;
                double affinity = controller.value("affinity", 0.0);
                std::string reply = base.value("reply", "");
                if (reply.empty())
                    continue;
                double latency = std::max(0.0, base.value("latency", 0.0));
                scoreAndVoteReply(reply, affinity, latency, 0.55, 0.25, 0.20, 1.0);
                gnnUsed++;
            }
        }

        if (transformerCandidates.is_array())
        {
            for (auto &candidate : transformerCandidates)
            {
                if (!candidate.is_object())
                    continue;
                std::string reply = candidate.value("reply", "");
                if (reply.empty())
                    continue;
                double latency = std::max(0.0, candidate.value("latency", 0.0));
                double affinity = candidate.contains("affinity") && candidate["affinity"].is_number()
                                      ? candidate["affinity"].get<double>()
                                      : cosineSim(requestEmbedding, textToMiniEmbedding(reply, 64));
                scoreAndVoteReply(reply, affinity, latency, 0.60, 0.20, 0.20, 1.10);
                transformerUsed++;
            }
        }

        std::vector<std::pair<std::string, double>> ranked;
        ranked.reserve(tokenVotes.size());
        for (auto &kv : tokenVotes)
            ranked.push_back(kv);
        std::sort(ranked.begin(), ranked.end(), [](const auto &a, const auto &b)
                  { return a.second > b.second; });
        if (ranked.size() > 24)
            ranked.resize(24);

        std::ostringstream ann;
        ann << "gnn_transformer_rf_stage3|rf=" << rfInstances_ << "|budget=" << voteBudget << "|n=" << used << "|gnn=" << gnnUsed << "|transformer=" << transformerUsed << ":";
        for (size_t i = 0; i < ranked.size(); i++)
        {
            if (i)
                ann << ' ';
            ann << ranked[i].first;
        }
        std::string annotation = ann.str();
        if (annotation.size() > 512)
            annotation.resize(512);

        double conf = 0.0;
        if (!rfScores.empty())
        {
            for (double s : rfScores)
                conf += s;
            conf /= (double)rfScores.size();
        }

        json topTokens = json::array();
        for (auto &p : ranked)
            topTokens.push_back(json{{"token", p.first}, {"vote", p.second}});

        return json{{"annotation", annotation},
                    {"confidence", conf},
                    {"reply", ""},
                    {"method", "gnn-transformer-rf-ensemble"},
                    {"rfInstances", rfInstances_},
                    {"voteBudget", voteBudget},
                    {"gnnSignals", gnnUsed},
                    {"transformerSignals", transformerUsed},
                    {"topTokens", topTokens}};
    }

    void recordHistory(const json &payload, const json &aggregate, const json &layerResults)
    {
        history_.push_back(json{{"ts", (int64_t)std::chrono::system_clock::now().time_since_epoch().count()}, {"request", json{{"text", payload.value("text", "")}, {"sessionId", payload.value("sessionId", "")}}}, {"aggregate", aggregate}, {"layers", layerResults}});
        if (history_.size() > 100)
            history_.erase(history_.begin());
    }
};

// ------------------ Learning ------------------

// ReinforcementLearner 执行基于测试语料的强化学习回路。
// 调用方式：外部触发 learn 后返回评估与调参结果。
// 实现思路：评估、聚类、统计后生成参数更新策略。
// 注意事项：测试集质量直接决定学习方向可靠性。
// 注意事项：learn 不保证单次迭代一定提升指标。
// 注意事项：调用时应避免与高峰推理强竞争资源。
class ReinforcementLearner : public IReinforcementLearner
{
public:
    ReinforcementLearner(std::shared_ptr<ControllerPoolBase> pool, fs::path testsDir)
        : pool_(std::move(pool)), testsDir_(std::move(testsDir)) {}

    json learn(int cycles) override
    {
        (void)cycles;
        auto files = listTestFiles();
        std::vector<json> evals;
        for (auto &file : files)
        {
            evals.push_back(evaluateOnce(file));
        }
        std::vector<std::vector<float>> features;
        for (auto &e : evals)
        {
            features.push_back(buildFeatureForDoc(e.value("reply", "")));
        }
        auto coords = reduceFeatures(features);
        auto clusters = cluster(coords, std::min(4, std::max(2, (int)std::sqrt((double)files.size()))));
        std::unordered_map<int, std::vector<json>> grouped;
        for (size_t i = 0; i < evals.size(); i++)
            grouped[clusters[i]].push_back(evals[i]);
        json clusterStats = json::array();
        for (auto &kv : grouped)
        {
            double sum = 0;
            for (auto &e : kv.second)
                sum += e.value("score", 0.0);
            clusterStats.push_back(json{{"gid", kv.first}, {"avg", kv.second.empty() ? 0.0 : sum / kv.second.size()}, {"n", (int)kv.second.size()}});
        }
        double baseAvg = 0;
        for (auto &e : evals)
            baseAvg += e.value("score", 0.0);
        baseAvg = evals.empty() ? 0.0 : baseAvg / evals.size();
        double learnedAvg = 0;
        for (auto &c : clusterStats)
            learnedAvg += c.value("avg", 0.0);
        learnedAvg = clusterStats.empty() ? 0.0 : learnedAvg / clusterStats.size();
        json evalStats{{"baseAvg", baseAvg}, {"learnedAvg", learnedAvg}, {"clusters", clusterStats}};
        auto params = adjustParams(evalStats);
        latest_ = json{{"ok", true}, {"evalStats", evalStats}, {"params", params}, {"evals", evals}, {"ts", (int64_t)std::chrono::system_clock::now().time_since_epoch().count()}};
        history_.push_back(latest_);
        return latest_;
    }

    json latest() const override { return history_.empty() ? json() : history_.back(); }

    json refreshTests(const fs::path &testsDir) override
    {
        testsDir_ = testsDir;
        return json{{"ok", true}, {"testsDir", testsDir_.string()}};
    }

    json setTestsDir(const fs::path &testsDir) override { return refreshTests(testsDir); }

private:
    std::shared_ptr<ControllerPoolBase> pool_;
    fs::path testsDir_;
    int maxDocs_{64};
    int topKWords_{30};
    double improvementThreshold_{0.01};
    int iterations_{5};
    bool useUmap_{true};
    std::vector<json> history_;
    json latest_ = json::object();

    std::vector<std::string> listTestFiles() const
    {
        std::vector<std::string> files;
        if (!fs::exists(testsDir_))
            return files;
        for (auto &e : fs::directory_iterator(testsDir_))
        {
            if (!e.is_regular_file())
                continue;
            auto ext = e.path().extension().string();
            std::string lower;
            for (char c : ext)
                lower.push_back((char)std::tolower((unsigned char)c));
            if (lower == ".txt")
                files.push_back(e.path().filename().string());
        }
        std::sort(files.begin(), files.end());
        if ((int)files.size() > maxDocs_)
            files.resize(maxDocs_);
        return files;
    }

    std::string readText(const std::string &file) const
    {
        return readTextFile(testsDir_ / file);
    }

    double scoreReplyQuality(const std::string &reply, const std::unordered_map<std::string, double> &seeds) const
    {
        auto words = tokenize(reply);
        if (words.empty())
            return 0.0;
        double coverage = 0.0;
        int count = 0;
        for (auto &kv : seeds)
        {
            auto linked = pool_->getActive()->runtime()->getMemeWords(kv.first);
            if (linked.empty())
                continue;
            std::unordered_set<std::string> linkedSet(linked.begin(), linked.end());
            int hit = 0;
            for (auto &w : words)
                if (linkedSet.count(w))
                    hit++;
            coverage += (double)hit / std::max(1.0, (double)linkedSet.size());
            count++;
        }
        coverage = count ? coverage / count : 0.0;
        double uniq = (double)std::unordered_set<std::string>(words.begin(), words.end()).size() / std::max(1.0, (double)words.size());
        return 0.7 * coverage + 0.3 * uniq;
    }

    std::vector<float> buildFeatureForDoc(const std::string &text) const
    {
        return textToMiniEmbedding(text, 128);
    }

    std::vector<std::array<double, 2>> reduceFeatures(const std::vector<std::vector<float>> &features) const
    {
        if (features.empty())
            return {};
        int nRows = (int)features.size();
        int nCols = (int)features[0].size();
        std::vector<double> mean(nCols, 0.0);
        for (auto &row : features)
            for (int j = 0; j < nCols; j++)
                mean[j] += row[j];
        for (int j = 0; j < nCols; j++)
            mean[j] /= std::max(1, nRows);
        std::vector<double> centered(nRows * nCols, 0.0);
        for (int i = 0; i < nRows; i++)
            for (int j = 0; j < nCols; j++)
                centered[i * nCols + j] = features[i][j] - mean[j];
        std::vector<std::vector<double>> cov(nCols, std::vector<double>(nCols, 0.0));
        for (int i = 0; i < nRows; i++)
        {
            for (int a = 0; a < nCols; a++)
            {
                for (int b = a; b < nCols; b++)
                {
                    double v = centered[i * nCols + a] * centered[i * nCols + b];
                    cov[a][b] += v;
                    if (a != b)
                        cov[b][a] += v;
                }
            }
        }
        auto powerIter = [&](std::vector<double> vec)
        {
            std::vector<double> next(nCols, 0.0);
            for (int i = 0; i < nCols; i++)
            {
                double acc = 0.0;
                for (int j = 0; j < nCols; j++)
                    acc += cov[i][j] * vec[j];
                next[i] = acc;
            }
            double norm = 0.0;
            for (double v : next)
                norm += v * v;
            norm = std::sqrt(norm) ? std::sqrt(norm) : 1.0;
            for (double &v : next)
                v /= norm;
            return next;
        };
        std::vector<double> v1(nCols, 0.0), v2(nCols, 0.0);
        for (int j = 0; j < nCols; j++)
            v1[j] = ((double)std::rand() / RAND_MAX) - 0.5;
        for (int i = 0; i < 8; i++)
            v1 = powerIter(v1);
        for (int j = 0; j < nCols; j++)
            v2[j] = ((double)std::rand() / RAND_MAX) - 0.5;
        for (int i = 0; i < 8; i++)
        {
            double proj = 0;
            for (int j = 0; j < nCols; j++)
                proj += v2[j] * v1[j];
            for (int j = 0; j < nCols; j++)
                v2[j] -= proj * v1[j];
            v2 = powerIter(v2);
        }
        std::vector<std::array<double, 2>> coords;
        coords.reserve(nRows);
        for (int i = 0; i < nRows; i++)
        {
            double x = 0, y = 0;
            for (int j = 0; j < nCols; j++)
            {
                double v = centered[i * nCols + j];
                x += v * v1[j];
                y += v * v2[j];
            }
            coords.push_back({x, y});
        }
        return coords;
    }

    std::vector<int> cluster(const std::vector<std::array<double, 2>> &coords, int k) const
    {
        auto out = clusterKMeans(coords, k);
        if (!out.empty())
            return out;
        if (coords.empty())
            return {};
        double minX = coords[0][0], maxX = coords[0][0];
        for (auto &c : coords)
        {
            minX = std::min(minX, c[0]);
            maxX = std::max(maxX, c[0]);
        }
        double step = (maxX - minX) / std::max(1, k);
        std::vector<int> fallback;
        fallback.reserve(coords.size());
        for (auto &c : coords)
        {
            int idx = (step <= 1e-6) ? 0 : (int)std::floor((c[0] - minX) / step);
            if (idx < 0)
                idx = 0;
            if (idx >= k)
                idx = k - 1;
            fallback.push_back(idx);
        }
        return fallback;
    }

    std::vector<int> clusterKMeans(const std::vector<std::array<double, 2>> &coords, int k) const
    {
        if (coords.empty() || k <= 0)
            return {};
        int n = (int)coords.size();
        k = std::min(k, n);
        std::mt19937 rng(0xC0FFEEu);
        std::uniform_int_distribution<int> pick(0, n - 1);

        std::vector<std::array<double, 2>> centroids;
        centroids.reserve(k);
        centroids.push_back(coords[pick(rng)]);
        for (int c = 1; c < k; c++)
        {
            std::vector<double> dist(n, 0.0);
            double total = 0.0;
            for (int i = 0; i < n; i++)
            {
                double best = std::numeric_limits<double>::max();
                for (auto &ct : centroids)
                {
                    double dx = coords[i][0] - ct[0];
                    double dy = coords[i][1] - ct[1];
                    double d = dx * dx + dy * dy;
                    if (d < best)
                        best = d;
                }
                dist[i] = best;
                total += best;
            }
            if (total <= 1e-9)
            {
                centroids.push_back(coords[pick(rng)]);
                continue;
            }
            std::uniform_real_distribution<double> ur(0.0, total);
            double r = ur(rng);
            double acc = 0.0;
            int idx = 0;
            for (int i = 0; i < n; i++)
            {
                acc += dist[i];
                if (acc >= r)
                {
                    idx = i;
                    break;
                }
            }
            centroids.push_back(coords[idx]);
        }

        std::vector<int> assign(n, 0);
        for (int iter = 0; iter < 25; iter++)
        {
            bool changed = false;
            for (int i = 0; i < n; i++)
            {
                int bestK = 0;
                double best = std::numeric_limits<double>::max();
                for (int c = 0; c < k; c++)
                {
                    double dx = coords[i][0] - centroids[c][0];
                    double dy = coords[i][1] - centroids[c][1];
                    double d = dx * dx + dy * dy;
                    if (d < best)
                    {
                        best = d;
                        bestK = c;
                    }
                }
                if (assign[i] != bestK)
                {
                    assign[i] = bestK;
                    changed = true;
                }
            }
            std::vector<std::array<double, 2>> next(k, {0.0, 0.0});
            std::vector<int> counts(k, 0);
            for (int i = 0; i < n; i++)
            {
                next[assign[i]][0] += coords[i][0];
                next[assign[i]][1] += coords[i][1];
                counts[assign[i]]++;
            }
            for (int c = 0; c < k; c++)
            {
                if (counts[c] == 0)
                {
                    next[c] = coords[pick(rng)];
                }
                else
                {
                    next[c][0] /= counts[c];
                    next[c][1] /= counts[c];
                }
            }
            centroids.swap(next);
            if (!changed)
                break;
        }
        return assign;
    }

    json adjustParams(const json &evalStats)
    {
        auto active = pool_->getActive();
        auto params = active->runtime()->cloneParams();
        double baseScore = evalStats.value("baseAvg", 0.0);
        double learnedScore = evalStats.value("learnedAvg", 0.0);
        if (learnedScore - baseScore >= improvementThreshold_)
        {
            params["iteration"] = std::min(12, params.value("iteration", 5) + 1);
            params["decayK"] = std::max(0.5, params.value("decayK", 1.0) - 0.05);
        }
        else
        {
            params["iteration"] = std::max(3, params.value("iteration", 5) - 1);
            params["decayK"] = std::min(2.0, params.value("decayK", 1.0) + 0.05);
        }
        active->applyParams(params);
        return params;
    }

    json evaluateOnce(const std::string &file)
    {
        std::string text = readText(file);
        auto ctrl = pool_->getActive();
        auto response = ctrl->respond(json{{"text", text}});
        std::unordered_map<std::string, double> seeds;
        for (auto &s : response.value("seeds", json::array()))
        {
            if (s.is_array() && s.size() >= 2)
                seeds[s[0].get<std::string>()] = s[1].get<double>();
        }
        double score = scoreReplyQuality(response.value("reply", ""), seeds);
        return json{{"file", file}, {"score", score}, {"words", tokenize(text)}, {"reply", response.value("reply", "")}};
    }
};

class AdversarialLearner : public IAdversarialLearner
{
public:
    explicit AdversarialLearner(std::shared_ptr<ControllerPoolBase> pool)
        : pool_(std::move(pool)), rng_(0x12345678u) {}

    json attackAndDefend(const json &samples) override
    {
        auto bench = samples.is_array() ? samples : json::array();
        if ((int)bench.size() > benchLimit_)
            bench.erase(bench.begin() + benchLimit_, bench.end());
        json rounds = json::array();
        for (int r = 0; r < attackRounds_; r++)
        {
            json results = json::array();
            for (auto &s : bench)
            {
                auto advs = generateAdversaries(s.get<std::string>());
                for (auto &a : advs)
                {
                    results.push_back(probe(a));
                }
            }
            rounds.push_back(json{{"phase", "attack"}, {"results", results}});
        }
        json fixes = json::array();
        for (int d = 0; d < defenseRounds_; d++)
        {
            for (auto &s : bench)
            {
                auto res = probe(s.get<std::string>());
                if (res.value("quality", 0.0) < 0.25)
                {
                    auto words = tokenize(res.value("reply", ""));
                    auto seeds = pool_->getActive()->runtime()->mapWordsToMemes(words);
                    std::vector<std::string> ids;
                    for (auto &kv : seeds)
                        ids.push_back(kv.first);
                    for (size_t i = 0; i + 1 < ids.size(); i++)
                    {
                        pool_->getActive()->runtime()->graphLink(ids[i], ids[i + 1], 0.5, 0);
                    }
                    fixes.push_back(json{{"sample", s}, {"addedEdges", (int)std::max<int>(0, (int)ids.size() - 1)}});
                }
            }
            rounds.push_back(json{{"phase", "defense"}, {"fixes", fixes}});
        }
        auto summary = summarize(rounds);
        latest_ = json{{"ok", true}, {"summary", summary}, {"rounds", rounds}};
        history_.push_back(latest_);
        return latest_;
    }

    json latest() const override { return history_.empty() ? json() : history_.back(); }

private:
    std::shared_ptr<ControllerPoolBase> pool_;
    int maxAdversaries_{64};
    double noiseLevel_{0.2};
    int attackRounds_{3};
    int defenseRounds_{3};
    int benchLimit_{50};
    Rng32 rng_;
    std::vector<json> history_;
    json latest_ = json::object();

    std::vector<std::string> perturbTokens(const std::vector<std::string> &tokens)
    {
        std::vector<std::string> out;
        for (auto &t : tokens)
        {
            double r = rng_.next();
            if (r < noiseLevel_ / 2)
                continue;
            out.push_back(t);
        }
        if (rng_.next() < noiseLevel_)
            out.push_back("noise_token_" + std::to_string((int)(rng_.next() * 1000)));
        return out;
    }

    std::vector<std::string> generateAdversaries(const std::string &text)
    {
        auto base = tokenize(text);
        std::vector<std::string> adversaries;
        int times = std::min(8, std::max(2, (int)(base.size() / 3)));
        for (int i = 0; i < times; i++)
        {
            auto mut = perturbTokens(base);
            std::ostringstream oss;
            for (size_t j = 0; j < mut.size(); j++)
            {
                if (j)
                    oss << " ";
                oss << mut[j];
            }
            adversaries.push_back(oss.str());
        }
        return adversaries;
    }

    json probe(const std::string &text)
    {
        auto ctrl = pool_->getActive();
        auto result = ctrl->respond(json{{"text", text}});
        std::unordered_map<std::string, double> seeds;
        for (auto &s : result.value("seeds", json::array()))
        {
            if (s.is_array() && s.size() >= 2)
                seeds[s[0].get<std::string>()] = s[1].get<double>();
        }
        double quality = qualityMetric(result.value("reply", ""), seeds);
        return json{{"text", text}, {"reply", result.value("reply", "")}, {"quality", quality}, {"latency", result.value("latency", 0)}, {"seeds", result.value("seeds", json::array())}};
    }

    double qualityMetric(const std::string &reply, const std::unordered_map<std::string, double> &seeds)
    {
        auto base = tokenize(reply);
        if (base.empty())
            return 0.0;
        double linkedHit = linkedCoverage(base, seeds);
        double uniq = (double)std::unordered_set<std::string>(base.begin(), base.end()).size() / std::max(1.0, (double)base.size());
        double lenPenalty = base.size() > 50 ? 0.9 : 1.0;
        return (0.6 * linkedHit + 0.4 * uniq) * lenPenalty;
    }

    double linkedCoverage(const std::vector<std::string> &words, const std::unordered_map<std::string, double> &seeds)
    {
        double coverage = 0.0;
        int count = 0;
        for (auto &kv : seeds)
        {
            auto linked = pool_->getActive()->runtime()->getMemeWords(kv.first);
            if (linked.empty())
                continue;
            std::unordered_set<std::string> linkedSet(linked.begin(), linked.end());
            int hit = 0;
            for (auto &w : words)
                if (linkedSet.count(w))
                    hit++;
            coverage += (double)hit / std::max(1.0, (double)linked.size());
            count++;
        }
        return count ? coverage / count : 0.0;
    }

    json summarize(const json &rounds)
    {
        json stats{{"attack", json{{"n", 0}, {"avgQuality", 0.0}}}, {"defense", json{{"fixes", 0}}}};
        double qsum = 0.0;
        int qcnt = 0;
        for (auto &r : rounds)
        {
            if (r.value("phase", "") == "attack")
            {
                auto results = r.value("results", json::array());
                stats["attack"]["n"] = stats["attack"]["n"].get<int>() + (int)results.size();
                for (auto &item : results)
                {
                    qsum += item.value("quality", 0.0);
                    qcnt++;
                }
            }
            else if (r.value("phase", "") == "defense")
            {
                stats["defense"]["fixes"] = stats["defense"]["fixes"].get<int>() + (int)r.value("fixes", json::array()).size();
            }
        }
        stats["attack"]["avgQuality"] = qcnt ? (qsum / qcnt) : 0.0;
        return stats;
    }
};

class GnnGaLearner : public IGnnGaLearner
{
public:
    GnnGaLearner(std::shared_ptr<ControllerPoolBase> pool, fs::path testsDir)
        : pool_(std::move(pool)), testsDir_(std::move(testsDir)) {}

    json evolve(int generations,
                int population,
                const std::vector<std::string> &samples,
                double residualWeight,
                double mutationRate,
                double mutationScale) override
    {
        generations = std::max(1, generations);
        population = std::max(4, population);
        residualWeight = std::max(0.0, std::min(1.0, residualWeight));
        mutationRate = std::max(0.0, std::min(1.0, mutationRate));
        mutationScale = std::max(0.05, std::min(0.75, mutationScale));

        auto texts = loadSamples(samples);
        if (texts.empty())
        {
            latest_ = json{{"ok", false}, {"error", "no-samples"}};
            history_.push_back(latest_);
            return latest_;
        }

        auto ctrl = pool_->getActive();
        if (!ctrl)
        {
            latest_ = json{{"ok", false}, {"error", "controller-not-available"}};
            history_.push_back(latest_);
            return latest_;
        }
        auto baseParams = ctrl->runtime()->cloneParams();
        Candidate base = fromParams(baseParams);

        std::vector<Candidate> pop;
        pop.reserve(population);
        pop.push_back(base);
        while ((int)pop.size() < population)
        {
            pop.push_back(mutate(base, mutationRate, mutationScale));
        }

        json genStats = json::array();
        Candidate best = base;
        for (int g = 0; g < generations; g++)
        {
            for (auto &c : pop)
            {
                c = evaluate(c, texts, residualWeight);
            }
            std::sort(pop.begin(), pop.end(), [](const Candidate &a, const Candidate &b)
                      { return a.fitness > b.fitness; });
            if (g == 0 || pop.front().fitness > best.fitness)
                best = pop.front();

            double avgFit = 0.0;
            for (auto &c : pop)
                avgFit += c.fitness;
            avgFit = pop.empty() ? 0.0 : avgFit / pop.size();
            genStats.push_back(json{{"gen", g}, {"best", pop.front().toJson()}, {"avgFitness", avgFit}});

            int elite = std::max(2, population / 4);
            std::vector<Candidate> next;
            next.reserve(population);
            for (int i = 0; i < elite && i < (int)pop.size(); i++)
                next.push_back(pop[i]);
            while ((int)next.size() < population)
            {
                const Candidate &a = pop[randIndex(elite)];
                const Candidate &b = pop[randIndex(elite)];
                Candidate child = crossover(a, b);
                child = mutate(child, mutationRate, mutationScale);
                next.push_back(child);
            }
            pop.swap(next);
        }

        ctrl->runtime()->setParams(patchForCandidate(best));
        latest_ = json{
            {"ok", true},
            {"best", best.toJson()},
            {"generations", generations},
            {"population", population},
            {"residualWeight", residualWeight},
            {"mutationRate", mutationRate},
            {"mutationScale", mutationScale},
            {"samples", (int)texts.size()},
            {"genStats", genStats},
            {"ts", (int64_t)std::chrono::system_clock::now().time_since_epoch().count()}};
        history_.push_back(latest_);
        return latest_;
    }

    json latest() const override { return history_.empty() ? json() : history_.back(); }

    json refreshTests(const fs::path &testsDir)
    {
        testsDir_ = testsDir;
        return json{{"ok", true}, {"testsDir", testsDir_.string()}};
    }

private:
    struct Candidate
    {
        int iteration{5};
        int maxLen{16};
        double decayK{1.0};
        double decayFactor{0.5};
        double threshold{3.0};
        double decay{1.0};
        double edgeWeight{1.0};
        double fitness{-1e9};
        double avgQuality{0.0};
        double avgResidual{0.0};

        json toJson() const
        {
            return json{{"iteration", iteration}, {"maxLen", maxLen}, {"decayK", decayK}, {"decayFactor", decayFactor}, {"threshold", threshold}, {"decay", decay}, {"edgeWeight", edgeWeight}, {"fitness", fitness}, {"avgQuality", avgQuality}, {"avgResidual", avgResidual}};
        }
    };

    std::shared_ptr<ControllerPoolBase> pool_;
    fs::path testsDir_;
    int maxDocs_{32};
    std::mt19937 rng_{0xC0FFEEu};
    std::vector<json> history_;
    json latest_ = json::object();

    int randIndex(int upper)
    {
        if (upper <= 1)
            return 0;
        std::uniform_int_distribution<int> dist(0, upper - 1);
        return dist(rng_);
    }

    static int clampIntRange(int v, int lo, int hi)
    {
        if (v < lo)
            return lo;
        if (v > hi)
            return hi;
        return v;
    }

    static double clampDoubleRange(double v, double lo, double hi)
    {
        if (!std::isfinite(v))
            return (lo + hi) * 0.5;
        if (v < lo)
            return lo;
        if (v > hi)
            return hi;
        return v;
    }

    Candidate fromParams(const json &params) const
    {
        Candidate c;
        c.iteration = clampIntRange(params.value("iteration", 5), 3, 12);
        c.maxLen = clampIntRange(params.value("maxLen", 16), 8, 32);
        c.decayK = clampDoubleRange(params.value("decayK", 1.0), 0.5, 2.0);
        c.decayFactor = clampDoubleRange(params.value("decayFactor", 0.5), 0.2, 0.9);
        c.threshold = clampDoubleRange(params.value("threshold", 3.0), 1.0, 6.0);
        c.decay = clampDoubleRange(params.value("decay", 1.0), 0.5, 2.0);
        c.edgeWeight = clampDoubleRange(params.value("edgeWeight", 1.0), 0.5, 2.0);
        return c;
    }

    json patchForCandidate(const Candidate &c) const
    {
        return json{{"iteration", c.iteration}, {"maxLen", c.maxLen}, {"decayK", c.decayK}, {"decayFactor", c.decayFactor}, {"threshold", c.threshold}, {"decay", c.decay}, {"edgeWeight", c.edgeWeight}};
    }

    Candidate crossover(const Candidate &a, const Candidate &b)
    {
        std::uniform_real_distribution<double> pick(0.0, 1.0);
        Candidate out;
        out.iteration = (pick(rng_) < 0.5) ? a.iteration : b.iteration;
        out.maxLen = (pick(rng_) < 0.5) ? a.maxLen : b.maxLen;
        out.decayK = (pick(rng_) < 0.5) ? a.decayK : b.decayK;
        out.decayFactor = (pick(rng_) < 0.5) ? a.decayFactor : b.decayFactor;
        out.threshold = (pick(rng_) < 0.5) ? a.threshold : b.threshold;
        out.decay = (pick(rng_) < 0.5) ? a.decay : b.decay;
        out.edgeWeight = (pick(rng_) < 0.5) ? a.edgeWeight : b.edgeWeight;
        return out;
    }

    Candidate mutate(Candidate c, double rate, double scale)
    {
        std::uniform_real_distribution<double> chance(0.0, 1.0);
        auto mutateDouble = [&](double v, double lo, double hi)
        {
            if (chance(rng_) >= rate)
                return v;
            double delta = (chance(rng_) * 2.0 - 1.0) * scale * (hi - lo);
            return clampDoubleRange(v + delta, lo, hi);
        };
        auto mutateInt = [&](int v, int lo, int hi)
        {
            if (chance(rng_) >= rate)
                return v;
            int span = std::max(1, hi - lo);
            int delta = (int)std::round(((chance(rng_) * 2.0 - 1.0) * scale * span));
            return clampIntRange(v + delta, lo, hi);
        };

        c.iteration = mutateInt(c.iteration, 3, 12);
        c.maxLen = mutateInt(c.maxLen, 8, 32);
        c.decayK = mutateDouble(c.decayK, 0.5, 2.0);
        c.decayFactor = mutateDouble(c.decayFactor, 0.2, 0.9);
        c.threshold = mutateDouble(c.threshold, 1.0, 6.0);
        c.decay = mutateDouble(c.decay, 0.5, 2.0);
        c.edgeWeight = mutateDouble(c.edgeWeight, 0.5, 2.0);
        return c;
    }

    Candidate evaluate(const Candidate &candidate, const std::vector<std::string> &texts, double residualWeight)
    {
        Candidate out = candidate;
        auto ctrl = pool_->getActive();
        if (!ctrl)
        {
            out.fitness = -1e9;
            return out;
        }
        auto runtime = ctrl->runtime();
        if (!runtime)
        {
            out.fitness = -1e9;
            return out;
        }

        json baseParams = runtime->cloneParams();
        runtime->setParams(patchForCandidate(candidate));
        double qsum = 0.0;
        double rsum = 0.0;
        int count = 0;
        try
        {
            for (const auto &text : texts)
            {
                auto resp = ctrl->respond(json{{"text", text}});
                std::unordered_map<std::string, double> seeds;
                for (auto &s : resp.value("seeds", json::array()))
                {
                    if (s.is_array() && s.size() >= 2 && s[0].is_string())
                        seeds[s[0].get<std::string>()] = s[1].get<double>();
                }
                std::string reply = resp.value("reply", "");
                qsum += scoreReplyQuality(reply, seeds);
                rsum += residualScore(text, reply);
                count++;
            }
        }
        catch (...)
        {
        }
        runtime->setParams(baseParams);

        out.avgQuality = count ? (qsum / count) : 0.0;
        out.avgResidual = count ? (rsum / count) : 1.0;
        out.fitness = out.avgQuality - residualWeight * out.avgResidual;
        return out;
    }

    double scoreReplyQuality(const std::string &reply, const std::unordered_map<std::string, double> &seeds) const
    {
        auto words = tokenize(reply);
        if (words.empty())
            return 0.0;
        double coverage = 0.0;
        int count = 0;
        for (auto &kv : seeds)
        {
            auto linked = pool_->getActive()->runtime()->getMemeWords(kv.first);
            if (linked.empty())
                continue;
            std::unordered_set<std::string> linkedSet(linked.begin(), linked.end());
            int hit = 0;
            for (auto &w : words)
                if (linkedSet.count(w))
                    hit++;
            coverage += (double)hit / std::max(1.0, (double)linkedSet.size());
            count++;
        }
        coverage = count ? coverage / count : 0.0;
        double uniq = (double)std::unordered_set<std::string>(words.begin(), words.end()).size() / std::max(1.0, (double)words.size());
        return 0.7 * coverage + 0.3 * uniq;
    }

    double residualScore(const std::string &question, const std::string &reply) const
    {
        if (question.empty() || reply.empty())
            return 1.0;
        auto qEmb = textToMiniEmbedding(question, 128);
        auto rEmb = textToMiniEmbedding(reply, 128);
        double cos = cosineSim(qEmb, rEmb);
        double sim = clamp01((cos + 1.0) * 0.5);
        return 1.0 - sim;
    }

    std::vector<std::string> loadSamples(const std::vector<std::string> &samples) const
    {
        if (!samples.empty())
            return samples;
        auto files = listTestFiles();
        std::vector<std::string> texts;
        for (auto &file : files)
        {
            auto text = readText(file);
            if (!text.empty())
                texts.push_back(text);
        }
        return texts;
    }

    std::vector<std::string> listTestFiles() const
    {
        std::vector<std::string> files;
        if (!fs::exists(testsDir_))
            return files;
        for (auto &e : fs::directory_iterator(testsDir_))
        {
            if (!e.is_regular_file())
                continue;
            auto ext = e.path().extension().string();
            std::string lower;
            for (char c : ext)
                lower.push_back((char)std::tolower((unsigned char)c));
            if (lower == ".txt")
                files.push_back(e.path().filename().string());
        }
        std::sort(files.begin(), files.end());
        if ((int)files.size() > maxDocs_)
            files.resize(maxDocs_);
        return files;
    }

    std::string readText(const std::string &file) const
    {
        return readTextFile(testsDir_ / file);
    }
};

// ------------------ Redis Sync ------------------

// RedisSynchronizer 负责跨实例快照同步广播。
// 调用方式：启动订阅线程并在切换时发布快照消息。
// 实现思路：基于 Redis Pub/Sub 传递 JSON 快照载荷。
// 注意事项：网络波动时需容忍短时同步延迟。
// 注意事项：订阅线程异常应避免影响主请求路径。
// 注意事项：该模块依赖 Redis 可用性。
class RedisSynchronizer : public IRedisSynchronizer
{
public:
    RedisSynchronizer(std::shared_ptr<ControllerPoolBase> pool, const std::string &url, const std::string &channel)
        : pool_(std::move(pool)), url_(url), channel_(channel) {}

    void start() override
    {
#ifdef HAVE_REDIS
        sw::redis::ConnectionOptions opts(url_);
        pub_ = std::make_unique<sw::redis::Redis>(opts);
        sub_ = std::make_unique<sw::redis::Redis>(opts);
        auto sub = sub_->subscriber();
        sub.subscribe(channel_);
        subThread_ = std::thread([this, sub = std::move(sub)]() mutable
                                 {
            while (running_) {
                sub.consume([&](std::string, std::string msg) {
                    try {
                        auto snapshot = json::parse(msg);
                        pool_->hotSwap(snapshot);
                    } catch (...) {}
                });
            } });
#endif
    }

    void publish(const json &snapshot) override
    {
#ifdef HAVE_REDIS
        if (pub_)
            pub_->publish(channel_, snapshot.dump());
#endif
    }

    const std::string &channel() const override { return channel_; }

private:
    std::shared_ptr<ControllerPoolBase> pool_;
    std::string url_;
    std::string channel_;
#ifdef HAVE_REDIS
    std::unique_ptr<sw::redis::Redis> pub_;
    std::unique_ptr<sw::redis::Redis> sub_;
#endif
    std::thread subThread_;
    std::atomic<bool> running_{true};
};

class RotationManager
{
public:
    RotationManager(std::shared_ptr<ControllerPoolBase> pool, int cycleMs = 5 * 60 * 1000, int learnIters = 3, double minImprove = 0.05)
        : pool_(std::move(pool)), cycleMs_(cycleMs), learnIters_(learnIters), minImprove_(minImprove) {}

    void start()
    {
        if (running_)
            return;
        running_ = true;
        worker_ = std::thread([this]()
                              {
            while (running_) {
                std::this_thread::sleep_for(std::chrono::milliseconds(cycleMs_));
                if (!running_) break;
                try { runCycle(); } catch (...) {}
            } });
    }

    void stop()
    {
        running_ = false;
        if (worker_.joinable())
            worker_.join();
    }

    bool running() const { return running_; }
    int cycleMs() const { return cycleMs_; }

private:
    double evaluateController(const std::shared_ptr<Controller> &ctrl, const std::vector<std::string> &prompts)
    {
        if (!ctrl)
            return -1e9;
        double quality = 0.0;
        int ok = 0;
        for (const auto &prompt : prompts)
        {
            try
            {
                auto out = ctrl->respond(json{{"text", prompt}});
                std::string reply = out.value("reply", "");
                if (reply.empty())
                    continue;
                auto q = tokenize(prompt);
                auto r = tokenize(reply);
                if (r.empty())
                    continue;
                std::unordered_set<std::string> rq(r.begin(), r.end());
                int hit = 0;
                for (auto &w : q)
                    if (rq.count(w))
                        hit++;
                double overlap = q.empty() ? 0.0 : (double)hit / std::max(1.0, (double)q.size());
                double diversity = (double)rq.size() / std::max(1.0, (double)r.size());
                quality += 0.65 * overlap + 0.35 * diversity;
                ok++;
            }
            catch (...)
            {
            }
        }
        if (ok == 0)
            return -1e9;
        return quality / ok;
    }

    void runCycle()
    {
        static const std::vector<std::string> prompts = {
            "请总结输入并给出关键结论", "解释这个问题的因果关系", "给出一个可执行的解决步骤",
            "将这段描述转换成结构化要点", "从风险和收益两个角度分析"};

        auto ctrlA = pool_->getActive();
        auto ctrlB = pool_->getStandby();
        auto ctrlC = pool_->getValidation();
        if (!ctrlA || !ctrlB || !ctrlC)
            return;

        double baseScore = evaluateController(ctrlA, prompts);
        double evalScore = evaluateController(ctrlB, prompts);

        if (!std::isfinite(baseScore) || !std::isfinite(evalScore))
            return;

        auto params = ctrlB->runtime()->cloneParams();
        int iteration = params.value("iteration", MODEL_DEFAULTS.iteration);
        iteration += std::max(1, learnIters_);
        params["iteration"] = iteration;
        ctrlB->applyParams(params);

        if (evalScore - baseScore >= minImprove_)
        {
            pool_->hotSwap(ctrlB->snapshot());
        }
        ctrlC->applySnapshot(ctrlB->snapshot());
    }

    std::shared_ptr<ControllerPoolBase> pool_;
    int cycleMs_{5 * 60 * 1000};
    int learnIters_{3};
    double minImprove_{0.05};
    std::atomic<bool> running_{false};
    std::thread worker_;
    std::mt19937 rng_{std::random_device{}()};
};

StudyEngine::StudyEngine(std::shared_ptr<ControllerPoolBase> pool, std::shared_ptr<IRedisSynchronizer> redis)
    : pool_(std::move(pool)), redis_(std::move(redis)) {}

void StudyEngine::start()
{
    if (running_)
        return;
    running_ = true;
    worker_ = std::thread([this]()
                          {
        while (running_) {
            std::this_thread::sleep_for(std::chrono::seconds(60));
            if (!running_) break;
            try { tick(); } catch (...) {}
        } });
}

void StudyEngine::enqueueDocument(const json &doc)
{
    std::lock_guard<std::mutex> lock(mu_);
    queue_.push_back(doc);
    metrics_["enqueued"] = metrics_.value("enqueued", 0) + 1;
}

json StudyEngine::status() const
{
    std::lock_guard<std::mutex> lock(mu_);
    return json{{"running", running_.load()}, {"queue", (int)queue_.size()}, {"metrics", metrics_}};
}

std::shared_ptr<LemmaWorkerPool> StudyEngine::ensureWorkerPool()
{
    std::lock_guard<std::mutex> lock(workerMu_);
    if (workerPool_)
        return workerPool_;
    int workers = 1;
    try
    {
        auto cfg = pool_ ? pool_->config() : Config{};
        workers = std::max(1, cfg.maxWorkers);
    }
    catch (...)
    {
        workers = 1;
    }
    workerPool_ = std::make_shared<LemmaWorkerPool>(workers);
    return workerPool_;
}

void StudyEngine::tick()
{
    json doc;
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (!running_ || queue_.empty())
            return;
        doc = queue_.front();
        queue_.pop_front();
    }
    try
    {
        {
            std::lock_guard<std::mutex> lock(mu_);
            metrics_["lastTickAt"] = (int64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
                                         std::chrono::system_clock::now().time_since_epoch())
                                         .count();
        }
        if (!doc.is_null())
        {
            pool_->ingestDocument(doc);
        }
        try
        {
            if (!doc.is_null())
            {
                std::string text = doc.value("text", "");
                auto tokens = tokenize(text);
                std::string lemmaCsv;
                if (pool_)
                {
                    auto active = pool_->getActive();
                    if (active)
                    {
                        try
                        {
                            lemmaCsv = active->runtime()->config().lemmaCsv.string();
                        }
                        catch (...)
                        {
                        }
                    }
                }
                auto wp = ensureWorkerPool();
                if (wp)
                {
                    wp->execBatchLemmatize(tokens, lemmaCsv);
                }
            }
        }
        catch (...)
        {
            // ignore worker errors
        }
        {
            std::lock_guard<std::mutex> lock(mu_);
            metrics_["processed"] = metrics_.value("processed", 0) + 1;
            metrics_["lastError"] = nullptr;
        }
    }
    catch (const std::exception &e)
    {
        std::lock_guard<std::mutex> lock(mu_);
        metrics_["lastError"] = std::string(e.what());
    }
    if (redis_)
    {
        try
        {
            auto snap = pool_->getStandby()->snapshot();
            redis_->publish(snap);
        }
        catch (...)
        {
        }
    }
}

// ------------------ Snapshot ------------------

// SnapshotManager 管理运行时快照的创建与恢复。
// 调用方式：网关路由调用 create/list/restore/remove。
// 实现思路：快照 JSON 原子写入并按数量进行清理。
// 注意事项：恢复时需校验快照结构兼容当前版本。
// 注意事项：文件系统异常会抛错并由上层处理。
// 注意事项：快照只覆盖运行态，不覆盖外部依赖。
class SnapshotManager : public ISnapshotManager
{
public:
    SnapshotManager(std::shared_ptr<RuntimeBase> runtime, fs::path dir)
        : runtime_(std::move(runtime)), dir_(std::move(dir))
    {
        ensureDir(dir_);
    }

    std::vector<std::string> list() const override
    {
        std::vector<std::string> out;
        if (!fs::exists(dir_))
            return out;
        for (auto &e : fs::directory_iterator(dir_))
        {
            if (e.path().extension() == ".json" && e.is_regular_file())
                out.push_back(e.path().filename().string());
        }
        std::sort(out.begin(), out.end(), std::greater<std::string>());
        return out;
    }

    fs::path create(const std::string &name = "auto") override
    {
        json snap = runtime_->toSnapshot();
        json doc = json::object();
        doc["meta"] = json{{"version", 1}, {"createdAt", nowIso()}, {"tag", name}};
        doc["snapshot"] = snap;
        auto ts = nowIso();
        std::string file = ts;
        std::replace(file.begin(), file.end(), ':', '-');
        std::replace(file.begin(), file.end(), '.', '-');
        file += "_" + name + ".json";
        fs::path out = dir_ / file;
        fs::path tmp = out;
        tmp += ".tmp";
        {
            std::ofstream o(tmp, std::ios::trunc);
            o << doc.dump(2);
        }
        if (fs::exists(out))
            fs::remove(out);
        fs::rename(tmp, out);

        auto files = list();
        const size_t maxKeep = 200;
        for (size_t i = maxKeep; i < files.size(); i++)
        {
            try
            {
                fs::remove(dir_ / files[i]);
            }
            catch (...)
            {
            }
        }
        return out;
    }

    json restore(const std::string &id) override
    {
        fs::path file = dir_ / fs::path(id).filename();
        if (!fs::exists(file))
            throw std::runtime_error("snapshot not found");
        std::ifstream in(file);
        json snap;
        in >> snap;
        if (snap.is_object() && snap.contains("snapshot"))
            snap = snap["snapshot"];
        runtime_->fromSnapshot(snap);
        return snap;
    }

    void remove(const std::string &id) override
    {
        fs::path file = dir_ / fs::path(id).filename();
        if (fs::exists(file))
        {
            fs::remove(file);
        }
    }

private:
    std::shared_ptr<RuntimeBase> runtime_;
    fs::path dir_;
};

// ------------------ Gateway ------------------

// GatewayServer 承载 HTTP 网关路由与系统编排逻辑。
// 调用方式：构造后调用 listen 启动对外服务。
// 实现思路：整合控制器池、学习器、快照与鉴权组件。
// 注意事项：路由较多，异常需统一捕获并标准化返回。
// 注意事项：内部模块初始化失败应尽量降级运行。
// 注意事项：该类是系统入口，变更需谨慎回归验证。
class GatewayServer : public IGatewayServer
{
public:
    GatewayServer(std::shared_ptr<ControllerPoolBase> pool,
                  std::shared_ptr<ISnapshotManager> snapshots,
                  std::shared_ptr<IRedisSynchronizer> redisSync,
                  std::shared_ptr<RotationManager> rotation,
                  std::shared_ptr<IStudyEngine> study,
                  std::shared_ptr<Database079> db079,
                  const Config &config)
        : pool_(std::move(pool)), snapshots_(std::move(snapshots)), redis_(std::move(redisSync)), rotation_(std::move(rotation)), study_(std::move(study)), db_(std::move(db079)), config_(config)
    {
        startedAt_ = std::chrono::steady_clock::now();
        shards_ = std::make_shared<ShardManager>(pool_);
        json sparkOptions;
        sparkOptions["groupIds"] = pool_->listGroupIds();
        sparkOptions["numAI"] = config_.sparkNumAI;
        sparkOptions["rfInstances"] = std::max(3, config_.sparkNumAI);
        sparkOptions["budget"] = config_.sparkBudget;
        auto sparkFactory = module_mount::ModuleRegistry::instance().sparkArrayFactory();
        if (sparkFactory)
            spark_ = sparkFactory(pool_, shards_, sparkOptions);
        else
            spark_ = std::make_shared<SparkArray>(pool_, shards_, sparkOptions);
        {
            std::string authFlag = getEnv("AI_AUTH_ENABLED", "true");
            std::transform(authFlag.begin(), authFlag.end(), authFlag.begin(), ::tolower);
            authEnabled_ = (authFlag != "false");
        }
        acceptLocalToken_ = boolFrom(getEnv("AI_AUTH_ACCEPT_LOCAL_TOKEN", "true"), true);
        jwtSecret_ = getEnv("AI_AUTH_JWT_SECRET", getEnv("AUTH_JWT_SECRET", "dev-secret-change-me"));
        rlDisabled_ = config_.disableLearning || config_.disableRL;
        advDisabled_ = config_.disableLearning || config_.disableADV;
        gnnGaDisabled_ = config_.disableLearning;
        dialogLearningEnabled_ = !(config_.disableLearning == true);
        dialogTransformerLearning_ = boolFrom(getEnv("AI_DIALOG_TRANSFORMER_LEARNING", "false"), false);
        dialogLearnGraphContext_ = boolFrom(getEnv("AI_DIALOG_LEARN_GRAPH_CONTEXT", "false"), false);
        retrainTrace_ = boolFrom(getEnv("AI_RETRAIN_TRACE", "false"), false);
        logger_.initialize(config_.logMode, config_.exportDir / "logs");
        if (logger_.enabled())
            logger_.log(LoggerCXX::Type::LOG, std::string("gateway init host=") + config_.gatewayHost + " port=" + std::to_string(config_.portGateway));
        matrixCache_ = std::make_shared<HotMatrixCache>(config_.redisUrl, config_.redisCacheDb, config_.redisCachePrefix + ":matrix");
        transformer::setAttentionCacheConfig(config_.redisUrl, config_.redisCacheDb, config_.redisCachePrefix + ":attn", 900, 3, 200000);
        if (!config_.disableBarrier)
        {
            try
            {
                auto rt = std::dynamic_pointer_cast<RuntimeState>(pool_->getActive()->runtime());
                if (rt)
                {
                    barrier_ = std::make_shared<MemeBarrier>(rt, 10000, MODEL_DEFAULTS.maliciousThreshold, 5);
                    barrier_->start();
                    if (logger_.enabled())
                        logger_.log(LoggerCXX::Type::MONITORING, "memebarrier started");
                }
            }
            catch (...)
            {
                barrier_.reset();
                if (logger_.enabled())
                    logger_.log(LoggerCXX::Type::WARNING, "memebarrier start failed");
            }
        }
        else if (logger_.enabled())
        {
            logger_.log(LoggerCXX::Type::MONITORING, "memebarrier disabled by config");
        }
        setupRoutes();
    }

    void listen() override
    {
        unsigned hw = std::thread::hardware_concurrency();
        unsigned defaultThreads = hw > 0 ? std::max(4u, std::min(16u, hw)) : 4u;
        int threadNum = static_cast<int>(numberOr(getEnv("AI_HTTP_THREADS", std::to_string(defaultThreads)), static_cast<long long>(defaultThreads)));
        if (threadNum < 2)
            threadNum = 2;
        drogon::app().setThreadNum(threadNum);
        drogon::app().addListener(config_.gatewayHost, config_.portGateway);
        drogon::app().run();
    }

    void warmupLearning() override
    {
        if (!config_.disableLearning && !config_.disableRL)
        {
            try
            {
                ensureRL()->learn(1);
            }
            catch (...)
            {
                rlDisabled_ = true;
            }
        }
        if (!config_.disableLearning && !config_.disableADV)
        {
            try
            {
                auto docs = pool_->getActive()->runtime()->collectRobotsDocuments(json{{"limit", 3}, {"shuffle", true}});
                json samples = json::array();
                for (auto &d : docs)
                {
                    if (!d.text.empty())
                        samples.push_back(d.text);
                    if (samples.size() >= 3)
                        break;
                }
                if (!samples.empty())
                    ensureADV()->attackAndDefend(samples);
            }
            catch (...)
            {
            }
        }

        if (!config_.disableLearning)
        {
            try
            {
                std::vector<transformer::TrainSample> samples;
                samples.reserve(160);

                auto isGoodText = [](const std::string &text)
                {
                    if (text.size() < 24)
                        return false;
                    int letters = 0;
                    int digits = 0;
                    int spaces = 0;
                    int punct = 0;
                    int controls = 0;
                    int cjk = 0;
                    for (unsigned char ch : text)
                    {
                        if (ch < 0x20 || ch == 0x7F)
                        {
                            controls++;
                            continue;
                        }
                        if (std::isspace(ch))
                        {
                            spaces++;
                            continue;
                        }
                        if (std::isalpha(ch))
                        {
                            letters++;
                            continue;
                        }
                        if (std::isdigit(ch))
                        {
                            digits++;
                            continue;
                        }
                        if (std::ispunct(ch))
                        {
                            punct++;
                            continue;
                        }
                        if (ch >= 0x80)
                            cjk++;
                    }
                    if (controls > 0)
                        return false;
                    int nonSpace = std::max(1, (int)text.size() - spaces);
                    double punctRatio = (double)punct / (double)nonSpace;
                    if (punctRatio > 0.45)
                        return false;
                    int semantic = letters + cjk;
                    if (semantic < std::max(10, (int)text.size() / 5))
                        return false;
                    return true;
                };

                auto addSample = [&](const std::string &raw)
                {
                    std::string text = trimCopy(raw);
                    if (!isGoodText(text))
                        return;
                    if (text.size() > 320)
                        text = text.substr(0, 320);
                    samples.push_back(transformer::TrainSample{text, text, ""});
                };

                try
                {
                    auto docs = pool_->getActive()->runtime()->collectRobotsDocuments(json{{"limit", 120}, {"shuffle", true}});
                    for (auto &d : docs)
                    {
                        addSample(d.text);
                        if (samples.size() >= 120)
                            break;
                    }
                }
                catch (...)
                {
                }

                try
                {
                    if (fs::exists(config_.testsDir))
                    {
                        for (auto &e : fs::directory_iterator(config_.testsDir))
                        {
                            if (!e.is_regular_file())
                                continue;
                            auto ext = e.path().extension().string();
                            std::string lowerExt;
                            lowerExt.reserve(ext.size());
                            for (char c : ext)
                                lowerExt.push_back((char)std::tolower((unsigned char)c));
                            if (lowerExt != ".txt")
                                continue;
                            std::string text = readTextFile(e.path());
                            if (text.empty())
                                continue;
                            std::istringstream iss(text);
                            std::string line;
                            while (std::getline(iss, line))
                            {
                                addSample(line);
                                if (samples.size() >= 180)
                                    break;
                            }
                            if (samples.size() >= 180)
                                break;
                        }
                    }
                }
                catch (...)
                {
                }

                if (!samples.empty())
                {
                    int epochs = std::max(1, std::min(3, (int)(samples.size() >= 120 ? 2 : 1)));
                    float lr = 0.0006f;
                    std::lock_guard<std::mutex> lock(transformerMu_);
                    transformer_.pretrain(samples, epochs, lr);
                }
            }
            catch (...)
            {
            }
        }
    }

    void bootstrapTransformerFromCorpus(int maxDocs = 48) override
    {
        if (config_.disableLearning)
            return;
        std::vector<transformer::TrainSample> samples;
        samples.reserve((size_t)std::max(8, maxDocs));
        auto isGoodText = [](const std::string &text)
        {
            if (text.size() < 24)
                return false;
            int letters = 0;
            int spaces = 0;
            int punct = 0;
            int controls = 0;
            int cjk = 0;
            for (unsigned char ch : text)
            {
                if (ch < 0x20 || ch == 0x7F)
                {
                    controls++;
                    continue;
                }
                if (std::isspace(ch))
                {
                    spaces++;
                    continue;
                }
                if (std::isalpha(ch))
                {
                    letters++;
                    continue;
                }
                if (std::ispunct(ch))
                {
                    punct++;
                    continue;
                }
                if (ch >= 0x80)
                    cjk++;
            }
            if (controls > 0)
                return false;
            int nonSpace = std::max(1, (int)text.size() - spaces);
            double punctRatio = (double)punct / (double)nonSpace;
            if (punctRatio > 0.45)
                return false;
            int semantic = letters + cjk;
            return semantic >= std::max(10, (int)text.size() / 6);
        };

        auto addSample = [&](const std::string &raw)
        {
            std::string text = trimCopy(raw);
            if (!isGoodText(text))
                return;
            if (text.size() > 280)
                text = text.substr(0, 280);
            samples.push_back(transformer::TrainSample{text, text, ""});
        };

        try
        {
            auto docs = pool_->getActive()->runtime()->collectRobotsDocuments(json{{"limit", std::max(8, maxDocs)}, {"shuffle", true}});
            for (auto &d : docs)
            {
                addSample(d.text);
                if ((int)samples.size() >= maxDocs)
                    break;
            }
        }
        catch (...)
        {
        }

        if (samples.size() < 8)
            return;

        try
        {
            std::lock_guard<std::mutex> lock(transformerMu_);
            transformer_.pretrain(samples, 1, 0.0008f);
        }
        catch (...)
        {
        }
    }

private:
    struct RouteMetric
    {
        uint64_t calls{0};
        uint64_t errors{0};
        uint64_t totalMs{0};
        uint64_t maxMs{0};
        uint64_t lastMs{0};
    };

    struct TrainMonitorState
    {
        uint64_t id{0};
        bool running{false};
        bool ok{false};
        std::string route;
        std::string stage;
        std::string error;
        int64_t startedAtMs{0};
        int64_t updatedAtMs{0};
        int64_t finishedAtMs{0};
        uint64_t rssBefore{0};
        uint64_t rssPeak{0};
        uint64_t rssAfter{0};
        uint64_t docs{0};
        uint64_t processedDocs{0};
        uint64_t samples{0};
        uint64_t chars{0};
        json requestSummary = json::object();
        json resultSummary = json::object();
    };

    std::shared_ptr<ControllerPoolBase> pool_;
    std::shared_ptr<ISnapshotManager> snapshots_;
    std::shared_ptr<IRedisSynchronizer> redis_;
    std::shared_ptr<RotationManager> rotation_;
    std::shared_ptr<IStudyEngine> study_;
    std::shared_ptr<Database079> db_;
    std::shared_ptr<HotMatrixCache> matrixCache_;
    std::shared_ptr<ShardManager> shards_;
    std::shared_ptr<ISparkArray> spark_;
    std::shared_ptr<IReinforcementLearner> rl_;
    std::shared_ptr<IAdversarialLearner> adv_;
    std::shared_ptr<IGnnGaLearner> gnnGa_;
    transformer::TransformerService transformer_;
    model_lifecycle::ModelLifecycleManager modelLifecycle_;
    autonomy::TransformerClusterManager clusterManager_;
    autonomy::SpiderAutonomyManager spiderAutonomy_;
    autonomy::OptimizerAutonomyManager optimizerAutonomy_;
    autonomy::DatasetCatalogManager datasetCatalog_;
    std::mutex transformerMu_;
    Config config_;
    std::chrono::steady_clock::time_point startedAt_;
    bool authEnabled_{true};
    bool acceptLocalToken_{true};
    std::string jwtSecret_;
    std::shared_ptr<MemeBarrier> barrier_;
    LoggerCXX &logger_{LoggerCXX::instance()};
    bool rlDisabled_{false};
    bool advDisabled_{false};
    bool gnnGaDisabled_{false};
    bool dialogLearningEnabled_{true};
    bool dialogTransformerLearning_{false};
    bool dialogLearnGraphContext_{false};
    bool dataCleaningEnabled_{true};
    bool retrainTrace_{false};
    size_t dataCleanMaxChars_{2048};
    std::atomic<uint64_t> cleanedInputCount_{0};
    std::atomic<uint64_t> cleanedSampleCount_{0};
    std::mutex routeMetricMu_;
    std::unordered_map<std::string, RouteMetric> routeMetrics_;
    std::mutex trainMonitorMu_;
    uint64_t trainMonitorSeq_{0};
    TrainMonitorState trainMonitorActive_;
    std::deque<json> trainMonitorHistory_;
    std::mutex fineTuningMu_;
    bool fineTuningRunning_{false};
    std::atomic<int> dialogAsyncInFlight_{0};
    int dialogAsyncLimit_{2};
    struct DialogCounters
    {
        int total{0};
        int lastRL{0};
        int lastADV{0};
        int lastGNN{0};
    } dialogCounters_;
    struct DialogThresholds
    {
        double rlEvery{20};
        double advEvery{30};
        double gnnEvery{40};
    } dialogThresholds_;
    bool releaseSnapshotLogged_{false};
    struct AuthError
    {
        std::string error;
        std::string message;
        drogon::HttpStatusCode code{drogon::k401Unauthorized};
    };
    static thread_local AuthError lastAuthError_;
    static thread_local bool hasAuthError_;

    template <typename Fn>
    void launchDialogTask(Fn &&fn)
    {
        int now = dialogAsyncInFlight_.load(std::memory_order_relaxed);
        if (now >= std::max(1, dialogAsyncLimit_))
            return;
        dialogAsyncInFlight_.fetch_add(1, std::memory_order_relaxed);
        std::thread([this, task = std::function<void()>(std::forward<Fn>(fn))]() mutable
                    {
            try { task(); } catch (...) {}
            dialogAsyncInFlight_.fetch_sub(1, std::memory_order_relaxed); })
            .detach();
    }

    json parseRequestBody(const drogon::HttpRequestPtr &req, bool &ok) const
    {
        ok = true;
        auto payload = req->getJsonObject();
        if (payload)
            return fromJsoncpp(*payload);
        auto body = req->getBody();
        if (body.empty())
            return json::object();
        auto contentType = req->getHeader("content-type");
        std::string lowered = contentType;
        std::transform(lowered.begin(), lowered.end(), lowered.begin(), ::tolower);
        auto tryParseJsonBody = [&]() -> std::optional<json>
        {
            try
            {
                return json::parse(std::string(body));
            }
            catch (...)
            {
                return std::nullopt;
            }
        };
        if (lowered.find("application/json") != std::string::npos)
        {
            auto parsed = tryParseJsonBody();
            if (parsed)
                return *parsed;
        }
        if (lowered.find("application/x-www-form-urlencoded") != std::string::npos)
        {
            return parseFormUrlEncoded(std::string(body));
        }
        std::string raw = std::string(body);
        auto l = raw.find_first_not_of(" \t\r\n");
        if (l != std::string::npos)
        {
            char c = raw[l];
            if (c == '{' || c == '[')
            {
                auto parsed = tryParseJsonBody();
                if (parsed)
                    return *parsed;
            }
        }
        ok = false;
        return json();
    }

    bool jsTruthy(const json &v) const
    {
        if (v.is_null())
            return false;
        if (v.is_boolean())
            return v.get<bool>();
        if (v.is_number())
            return v.get<double>() != 0.0;
        if (v.is_string())
            return !v.get<std::string>().empty();
        return true;
    }

    struct AddonToolCall
    {
        bool requested{false};
        std::string type;
        std::string input;
    };

    static std::string trimText(std::string s)
    {
        auto start = s.find_first_not_of(" \t\r\n");
        if (start == std::string::npos)
            return "";
        auto end = s.find_last_not_of(" \t\r\n");
        return s.substr(start, end - start + 1);
    }

    std::string sanitizeInputText(const std::string &raw, bool &changed) const
    {
        changed = false;
        if (!dataCleaningEnabled_)
            return raw;
        std::string out;
        out.reserve(raw.size());
        bool previousSpace = false;
        for (unsigned char ch : raw)
        {
            if (ch < 0x20 || ch == 0x7F)
            {
                if (ch == '\t' || ch == '\n' || ch == '\r')
                {
                    if (!previousSpace)
                    {
                        out.push_back(' ');
                        previousSpace = true;
                    }
                }
                changed = true;
                continue;
            }
            if (std::isspace(ch))
            {
                if (!previousSpace)
                {
                    out.push_back(' ');
                    previousSpace = true;
                }
                continue;
            }
            out.push_back((char)ch);
            previousSpace = false;
        }
        auto trimmed = trimText(out);
        if (trimmed.size() > dataCleanMaxChars_)
        {
            trimmed.resize(dataCleanMaxChars_);
            changed = true;
        }
        if (trimmed != raw)
            changed = true;
        return trimmed;
    }

    static std::string attachAddonToolContract(const std::string &graphContext)
    {
        std::string out = graphContext;
        if (!out.empty())
            out += "\n";
        out += "[tool_contract] For arithmetic/calculation tasks, emit: [[ADDON:math|<expression>]] as the first line; for web/online fact lookup tasks, emit: [[ADDON:search|<query>]] as the first line. Keep payload plain and executable. [/tool_contract]";
        return out;
    }

    static AddonToolCall parseAddonToolCall(const std::string &reply, const std::string &fallbackInput)
    {
        AddonToolCall out;
        try
        {
            std::regex re(R"(\[\[ADDON:([a-zA-Z0-9_\-]+)\|(.+?)\]\])", std::regex::icase);
            std::smatch m;
            if (std::regex_search(reply, m, re) && m.size() >= 3)
            {
                out.requested = true;
                out.type = trimText(m[1].str());
                out.input = trimText(m[2].str());
                if (out.input.empty())
                    out.input = trimText(fallbackInput);
            }
        }
        catch (...)
        {
            out = AddonToolCall{};
        }
        return out;
    }

    static std::string stripAddonToolDirective(const std::string &reply)
    {
        try
        {
            std::regex re(R"(\s*\[\[ADDON:[^\]]+\]\]\s*)", std::regex::icase);
            return trimText(std::regex_replace(reply, re, " "));
        }
        catch (...)
        {
            return trimText(reply);
        }
    }

    static std::string buildAddonResultContext(const std::string &graphContext,
                                               const AddonToolCall &call,
                                               const addon::AddonResult &result)
    {
        std::string out = graphContext;
        if (!out.empty())
            out += "\n";
        std::string addonType = call.type.empty() ? result.meta.value("addon", std::string("unknown")) : call.type;
        out += "[tool_result addon=" + addonType + "]";
        if (!call.input.empty())
            out += " input=" + call.input;
        out += " output=" + result.reply;
        out += " [/tool_result]";
        return out;
    }

    bool invokeAddonTool(const AddonToolCall &call, const json &payload, addon::AddonResult &result) const
    {
        result = addon::AddonResult{};
        if (!call.requested)
            return false;
        std::string typeLower = call.type;
        std::transform(typeLower.begin(), typeLower.end(), typeLower.begin(), [](unsigned char c)
                       { return (char)std::tolower(c); });
        if (!typeLower.empty() && typeLower != "math" && typeLower != "search" && typeLower != "research" && typeLower != "web")
            return false;
        auto controller = pool_->getActive();
        if (!controller)
            return false;
        auto runtime = std::dynamic_pointer_cast<RuntimeState>(controller->runtime());
        if (!runtime)
            return false;
        json toolPayload = payload;
        if (!typeLower.empty())
            toolPayload["__addonType"] = typeLower;
        result = runtime->runAddonTool(call.input, toolPayload);
        if (result.handled && logger_.enabled())
        {
            logger_.log(LoggerCXX::Type::COMPUTE,
                        std::string("addon_tool handled type=") + (typeLower.empty() ? "auto" : typeLower) +
                            " input=" + call.input +
                            " addon=" + result.meta.value("addon", std::string("unknown")));
        }
        return result.handled;
    }

    bool isGroupProc() const { return dynamic_cast<GroupProcessPool *>(pool_.get()) != nullptr; }

    bool authOK(const drogon::HttpRequestPtr &req)
    {
        hasAuthError_ = false;
        if (!authEnabled_)
            return true;
        if (req->method() == drogon::Options)
            return true;
        auto path = req->path();
        if (path == "/api/system/status")
            return true;
        if (path.rfind("/api/", 0) != 0)
            return true;
        auto auth = req->getHeader("authorization");
        std::string token;
        try
        {
            std::regex re("^Bearer\\s+(.+)$", std::regex::icase);
            std::smatch m;
            if (std::regex_match(auth, m, re) && m.size() >= 2)
            {
                token = m[1].str();
            }
        }
        catch (...)
        {
            token.clear();
        }
        if (token.empty())
        {
            lastAuthError_ = {"unauthorized", "", drogon::k401Unauthorized};
            hasAuthError_ = true;
            logSecurityEvent(std::string("auth failed path=") + path + " reason=missing-token");
            return false;
        }
        if (acceptLocalToken_ && token.rfind("local-", 0) == 0)
        {
            return true;
        }
        try
        {
            auto dec = jwt::decode<jwt::traits::nlohmann_json>(token);
            jwt::verify<jwt::traits::nlohmann_json>()
                .allow_algorithm(jwt::algorithm::hs256{jwtSecret_})
                .verify(dec);
            return true;
        }
        catch (const std::exception &e)
        {
            lastAuthError_ = {"invalid-token", e.what(), drogon::k401Unauthorized};
            hasAuthError_ = true;
            logSecurityEvent(std::string("auth failed path=") + path + " reason=" + e.what());
            return false;
        }
        catch (...)
        {
            lastAuthError_ = {"invalid-token", "invalid token", drogon::k401Unauthorized};
            hasAuthError_ = true;
            logSecurityEvent(std::string("auth failed path=") + path + " reason=invalid-token");
            return false;
        }
    }

    json chatWithOllama(const std::string &text, const std::string &graphContext, int maxTokens)
    {
        json out = json::object();
        out["provider"] = config_.ollamaFineTuning ? "ollama+fine-tuning" : "ollama";
        try
        {
            auto pickAvailableOllamaModel = [&]() -> std::string {
                try {
                    auto client = drogon::HttpClient::newHttpClient(config_.ollamaBaseUrl.empty() ? std::string("http://127.0.0.1:11434") : config_.ollamaBaseUrl);
                    if (!client)
                        return "";
                    auto tagsReq = drogon::HttpRequest::newHttpRequest();
                    tagsReq->setMethod(drogon::Get);
                    tagsReq->setPath("/api/tags");
                    std::promise<std::pair<drogon::ReqResult, drogon::HttpResponsePtr>> tagsPromise;
                    auto tagsFuture = tagsPromise.get_future();
                    client->sendRequest(tagsReq, [&tagsPromise](drogon::ReqResult result, const drogon::HttpResponsePtr &resp)
                                        {
                    try
                    {
                        tagsPromise.set_value({result, resp});
                    }
                    catch (...)
                    {
                    } });
                    int waitMs = std::max(1000, std::min(config_.ollamaRequestTimeoutMs, 10000));
                    if (tagsFuture.wait_for(std::chrono::milliseconds(waitMs)) != std::future_status::ready)
                        return "";
                    auto tagsPair = tagsFuture.get();
                    if (tagsPair.first != drogon::ReqResult::Ok || !tagsPair.second)
                        return "";
                    if (tagsPair.second->statusCode() < 200 || tagsPair.second->statusCode() >= 300)
                        return "";
                    json tagsDoc = json::parse(tagsPair.second->getBody(), nullptr, false);
                    if (tagsDoc.is_discarded() || !tagsDoc.contains("models") || !tagsDoc["models"].is_array())
                        return "";
                    for (const auto &m : tagsDoc["models"]) {
                        if (!m.is_object())
                            continue;
                        std::string name = trimCopy(m.value("name", ""));
                        if (!name.empty())
                            return name;
                        std::string model = trimCopy(m.value("model", ""));
                        if (!model.empty())
                            return model;
                    }
                } catch (...) {
                }
                return "";
            };

            std::string selectedModel = trimCopy(config_.ollamaModel);
            if (selectedModel.empty()) {
                selectedModel = pickAvailableOllamaModel();
            }
            if (selectedModel.empty()) {
                out["ok"] = false;
                out["error"] = "ollama model not available";
                out["reply"] = "";
                return out;
            }
            out["model"] = selectedModel;

            std::string prompt = text;
            if (!graphContext.empty())
            {
                prompt = std::string("Context:\n") + graphContext + "\n\nUser:\n" + text;
            }
            json payload = {
                {"model", selectedModel},
                {"stream", false},
                {"messages", json::array({json{{"role", "user"}, {"content", prompt}}})},
                {"options", json{{"num_predict", std::max(16, maxTokens)}}}};

            auto client = drogon::HttpClient::newHttpClient(config_.ollamaBaseUrl.empty() ? std::string("http://127.0.0.1:11434") : config_.ollamaBaseUrl);
            if (!client)
            {
                out["ok"] = false;
                out["error"] = "ollama client init failed";
                out["reply"] = "";
                return out;
            }

            auto req = drogon::HttpRequest::newHttpRequest();
            req->setMethod(drogon::Post);
            req->setPath("/api/chat");
            req->setContentTypeCode(drogon::CT_APPLICATION_JSON);
            req->setBody(payload.dump());

            std::promise<std::pair<drogon::ReqResult, drogon::HttpResponsePtr>> promise;
            auto future = promise.get_future();
            client->sendRequest(req, [&promise](drogon::ReqResult result, const drogon::HttpResponsePtr &resp)
                                {
                try
                {
                    promise.set_value({result, resp});
                }
                catch (...)
                {
                } });

            if (future.wait_for(std::chrono::milliseconds(config_.ollamaRequestTimeoutMs)) != std::future_status::ready)
            {
                out["ok"] = false;
                out["error"] = "ollama request timeout";
                out["timeoutMs"] = config_.ollamaRequestTimeoutMs;
                out["reply"] = "";
                return out;
            }

            auto pair = future.get();
            auto reqResult = pair.first;
            auto resp = pair.second;
            if (reqResult != drogon::ReqResult::Ok || !resp)
            {
                out["ok"] = false;
                out["error"] = "ollama request failed";
                out["reply"] = "";
                return out;
            }
            if (resp->statusCode() < 200 || resp->statusCode() >= 300)
            {
                out["ok"] = false;
                out["status"] = (int)resp->statusCode();
                out["error"] = "ollama bad status";
                out["reply"] = "";
                return out;
            }

            json doc = json::parse(resp->getBody(), nullptr, false);
            if (doc.is_discarded())
            {
                out["ok"] = false;
                out["error"] = "ollama invalid json";
                out["reply"] = "";
                return out;
            }

            std::string reply;
            if (doc.contains("message") && doc["message"].is_object() && doc["message"].contains("content") && doc["message"]["content"].is_string())
            {
                reply = doc["message"]["content"].get<std::string>();
            }
            else if (doc.contains("response") && doc["response"].is_string())
            {
                reply = doc["response"].get<std::string>();
            }

            out["ok"] = !reply.empty();
            out["reply"] = reply;
            if (doc.contains("done"))
                out["done"] = doc["done"];
            if (doc.contains("total_duration"))
                out["total_duration"] = doc["total_duration"];
            return out;
        }
        catch (const std::exception &e)
        {
            out["ok"] = false;
            out["error"] = e.what();
            out["reply"] = "";
            return out;
        }
        catch (...)
        {
            out["ok"] = false;
            out["error"] = "ollama unknown error";
            out["reply"] = "";
            return out;
        }
    }

    json chatWithExternalAdapter(const std::string &provider,
                                 const std::string &baseUrl,
                                 const std::string &model,
                                 int timeoutMs,
                                 const std::string &text,
                                 const std::string &graphContext,
                                 int maxTokens)
    {
        json out = json::object();
        out["provider"] = provider;
        try
        {
            std::string selectedModel = trimCopy(model);
            if (selectedModel.empty())
                selectedModel = provider;
            out["model"] = selectedModel;

            std::string prompt = text;
            if (!graphContext.empty())
                prompt = std::string("Context:\n") + graphContext + "\n\nUser:\n" + text;

            json payload = {
                {"model", selectedModel},
                {"stream", false},
                {"messages", json::array({json{{"role", "user"}, {"content", prompt}}})},
                {"options", json{{"num_predict", std::max(16, maxTokens)}}}};

            auto endpoint = trimCopy(baseUrl);
            if (endpoint.empty())
                endpoint = "http://127.0.0.1:8080";
            auto client = drogon::HttpClient::newHttpClient(endpoint);
            if (!client)
            {
                out["ok"] = false;
                out["error"] = provider + " client init failed";
                out["reply"] = "";
                return out;
            }

            auto req = drogon::HttpRequest::newHttpRequest();
            req->setMethod(drogon::Post);
            req->setPath("/api/chat");
            req->setContentTypeCode(drogon::CT_APPLICATION_JSON);
            req->setBody(payload.dump());

            std::promise<std::pair<drogon::ReqResult, drogon::HttpResponsePtr>> promise;
            auto future = promise.get_future();
            client->sendRequest(req, [&promise](drogon::ReqResult result, const drogon::HttpResponsePtr &resp)
                                {
                try
                {
                    promise.set_value({result, resp});
                }
                catch (...)
                {
                } });

            if (future.wait_for(std::chrono::milliseconds(std::max(1000, timeoutMs))) != std::future_status::ready)
            {
                out["ok"] = false;
                out["error"] = provider + " request timeout";
                out["timeoutMs"] = std::max(1000, timeoutMs);
                out["reply"] = "";
                return out;
            }

            auto pair = future.get();
            auto reqResult = pair.first;
            auto resp = pair.second;
            if (reqResult != drogon::ReqResult::Ok || !resp)
            {
                out["ok"] = false;
                out["error"] = provider + " request failed";
                out["reply"] = "";
                return out;
            }
            if (resp->statusCode() < 200 || resp->statusCode() >= 300)
            {
                out["ok"] = false;
                out["status"] = (int)resp->statusCode();
                out["error"] = provider + " bad status";
                out["reply"] = "";
                return out;
            }

            json doc = json::parse(resp->getBody(), nullptr, false);
            if (doc.is_discarded())
            {
                out["ok"] = false;
                out["error"] = provider + " invalid json";
                out["reply"] = "";
                return out;
            }

            std::string reply;
            if (doc.contains("message") && doc["message"].is_object() && doc["message"].contains("content") && doc["message"]["content"].is_string())
                reply = doc["message"]["content"].get<std::string>();
            else if (doc.contains("response") && doc["response"].is_string())
                reply = doc["response"].get<std::string>();

            out["ok"] = !reply.empty();
            out["reply"] = reply;
            return out;
        }
        catch (const std::exception &e)
        {
            out["ok"] = false;
            out["error"] = e.what();
            out["reply"] = "";
            return out;
        }
        catch (...)
        {
            out["ok"] = false;
            out["error"] = provider + " unknown error";
            out["reply"] = "";
            return out;
        }
    }

    json chatWithLlamaCpp(const std::string &text, const std::string &graphContext, int maxTokens)
    {
        return chatWithExternalAdapter("llamacpp", config_.llamaCppBaseUrl, config_.llamaCppModel, config_.llamaCppRequestTimeoutMs, text, graphContext, maxTokens);
    }

    json chatWithBitNet(const std::string &text, const std::string &graphContext, int maxTokens)
    {
        return chatWithExternalAdapter("bitnet", config_.bitnetBaseUrl, config_.bitnetModel, config_.bitnetRequestTimeoutMs, text, graphContext, maxTokens);
    }

    struct ContextHint
    {
        std::string text;
        std::string mode;
        double weight{0.35};
    };

    std::shared_ptr<RuntimeBase> activeRuntime() const
    {
        auto ctrl = pool_ ? pool_->getActive() : nullptr;
        return ctrl ? ctrl->runtime() : nullptr;
    }

    std::string buildGraphContext(const json &graphResult, const json &payload)
    {
        auto runtime = activeRuntime();
        if (!runtime)
            return "";
        std::vector<std::string> words;
        if (payload.contains("tokens") && payload["tokens"].is_array())
        {
            for (auto &t : payload["tokens"])
            {
                if (t.is_string())
                    words.push_back(t.get<std::string>());
            }
        }
        else if (payload.contains("text") && payload["text"].is_string())
        {
            words = tokenize(payload["text"].get<std::string>());
        }

        std::vector<std::pair<std::string, double>> memes;
        std::unordered_set<std::string> seedIds;
        if (graphResult.contains("seeds") && graphResult["seeds"].is_array())
        {
            for (auto &s : graphResult["seeds"])
            {
                if (s.is_array() && s.size() >= 1 && s[0].is_string())
                    seedIds.insert(s[0].get<std::string>());
            }
        }
        if (graphResult.contains("memes") && graphResult["memes"].is_array())
        {
            auto mem = graphResult["memes"];
            auto act = graphResult.value("activation", json::array());
            for (size_t i = 0; i < mem.size(); i++)
            {
                if (!mem[i].is_string())
                    continue;
                double score = 0.0;
                if (act.is_array() && i < act.size() && act[i].is_number())
                    score = act[i].get<double>();
                memes.push_back({mem[i].get<std::string>(), score});
            }
        }
        std::sort(memes.begin(), memes.end(), [](auto &a, auto &b)
                  { return a.second > b.second; });
        if (memes.size() > 16)
            memes.resize(16);
        std::unordered_set<std::string> memeSet;
        for (auto &m : memes)
            memeSet.insert(m.first);

        std::ostringstream oss;
        oss << "ctx_v2";
        if (!words.empty())
        {
            oss << "\ninput:";
            for (size_t i = 0; i < words.size() && i < 24; i++)
            {
                if (i)
                    oss << " ";
                oss << words[i];
            }
        }
        if (!seedIds.empty())
        {
            oss << "\nseeds:";
            int cnt = 0;
            for (auto &sid : seedIds)
            {
                if (cnt++)
                    oss << ",";
                oss << sid;
                if (cnt >= 16)
                    break;
            }
        }

        oss << "\nnodes:";
        int nodeCount = 0;
        auto rtState = std::dynamic_pointer_cast<RuntimeState>(runtime);
        for (auto &m : memes)
        {
            if (nodeCount++)
                oss << " ; ";
            auto ws = runtime->getMemeWords(m.first);
            std::string kws;
            int kwCount = 0;
            for (auto &w : ws)
            {
                if (w.empty())
                    continue;
                if (kwCount++)
                    kws += ",";
                kws += w;
                if (kwCount >= 4)
                    break;
            }
            oss << m.first
                << "|act=" << std::fixed << std::setprecision(3) << m.second
                << "|deg=" << (rtState ? rtState->graph().degreeOf(m.first) : (int)ws.size())
                << "|seed=" << (seedIds.count(m.first) ? 1 : 0)
                << "|kw=" << kws;
            if (nodeCount >= 12)
                break;
        }

        oss << "\nedges:";
        int edgeCount = 0;
        if (rtState)
        {
            for (auto &m : memes)
            {
                for (auto &n : rtState->graph().neighborsOf(m.first))
                {
                    if (!memeSet.count(n.first))
                        continue;
                    if (edgeCount++)
                        oss << " ; ";
                    oss << m.first << "->" << n.first
                        << "|w=" << std::fixed << std::setprecision(2) << n.second.first
                        << "|d=" << n.second.second;
                    if (edgeCount >= 20)
                        break;
                }
                if (edgeCount >= 20)
                    break;
            }
        }
        else
        {
            for (size_t i = 0; i + 1 < memes.size() && edgeCount < 20; i++)
            {
                if (edgeCount++)
                    oss << " ; ";
                oss << memes[i].first << "->" << memes[i + 1].first << "|w=0.35|d=0";
            }
        }

        oss << "\nsummary:n=" << memes.size() << " e=" << edgeCount;
        std::string out = oss.str();
        if (out.size() > 4096)
            out.resize(4096);
        return out;
    }

    struct GnnHintSummary
    {
        double align{0.0};
        std::vector<std::string> keywords;
        std::string annotation;
    };

    static double tokenOverlapJaccard(const std::vector<std::string> &a, const std::vector<std::string> &b)
    {
        if (a.empty() || b.empty())
            return 0.0;
        std::unordered_set<std::string> sa(a.begin(), a.end());
        std::unordered_set<std::string> sb(b.begin(), b.end());
        double inter = 0.0;
        for (auto &t : sa)
            if (sb.count(t))
                inter++;
        double uni = (double)sa.size() + (double)sb.size() - inter;
        return uni <= 0.0 ? 0.0 : (inter / uni);
    }

    GnnHintSummary buildGnnHintSummary(const json &graphResult, const json &payload, const ContextHint &hint)
    {
        GnnHintSummary out;
        auto runtime = activeRuntime();
        if (!runtime)
            return out;

        std::vector<std::pair<std::string, double>> memes;
        if (graphResult.contains("memes") && graphResult["memes"].is_array())
        {
            auto mem = graphResult["memes"];
            auto act = graphResult.value("activation", json::array());
            for (size_t i = 0; i < mem.size(); i++)
            {
                if (!mem[i].is_string())
                    continue;
                double score = 0.0;
                if (act.is_array() && i < act.size() && act[i].is_number())
                    score = act[i].get<double>();
                memes.push_back({mem[i].get<std::string>(), score});
            }
        }
        std::sort(memes.begin(), memes.end(), [](auto &a, auto &b)
                  { return a.second > b.second; });
        if (memes.size() > 8)
            memes.resize(8);

        std::unordered_set<std::string> seen;
        for (auto &m : memes)
        {
            auto ws = runtime->getMemeWords(m.first);
            for (auto &w : ws)
            {
                if (w.empty())
                    continue;
                for (auto &t : tokenize(w))
                {
                    if (t.empty())
                        continue;
                    if (seen.insert(t).second)
                        out.keywords.push_back(t);
                    if (out.keywords.size() >= 24)
                        break;
                }
                if (out.keywords.size() >= 24)
                    break;
            }
            if (out.keywords.size() >= 24)
                break;
        }

        std::vector<std::string> hintTokens;
        if (!hint.text.empty())
        {
            hintTokens = tokenize(hint.text);
        }
        else if (payload.contains("tokens") && payload["tokens"].is_array())
        {
            for (auto &t : payload["tokens"])
                if (t.is_string())
                    hintTokens.push_back(t.get<std::string>());
        }
        else if (payload.contains("text") && payload["text"].is_string())
        {
            hintTokens = tokenize(payload["text"].get<std::string>());
        }
        out.align = tokenOverlapJaccard(out.keywords, hintTokens);

        if (!out.keywords.empty())
        {
            double w = std::max(0.15, std::min(1.0, 0.2 + 0.8 * out.align));
            std::ostringstream oss;
            oss << "gnn_stage2|align=" << std::fixed << std::setprecision(2) << out.align << "|w=" << std::fixed << std::setprecision(2) << w << ":";
            for (size_t i = 0; i < out.keywords.size(); i++)
            {
                if (i)
                    oss << " ";
                oss << out.keywords[i];
            }
            out.annotation = oss.str();
            if (out.annotation.size() > 512)
                out.annotation.resize(512);
        }
        return out;
    }

    ContextHint harmonizeHintWithGnn(const ContextHint &hint, double align) const
    {
        ContextHint out = hint;
        if (out.text.empty())
            return out;
        double gate = std::max(0.2, std::min(1.0, 0.4 + 0.6 * align));
        out.weight = clamp01(out.weight * gate);
        return out;
    }

    ContextHint extractContextHint(const json &payload) const
    {
        ContextHint hint;
        if (payload.contains("contextHint") && payload["contextHint"].is_string())
        {
            hint.text = payload["contextHint"].get<std::string>();
        }
        else if (payload.contains("context") && payload["context"].is_string())
        {
            hint.text = payload["context"].get<std::string>();
        }
        if (payload.contains("contextMode") && payload["contextMode"].is_string())
        {
            hint.mode = payload["contextMode"].get<std::string>();
        }
        if (payload.contains("contextWeight") && payload["contextWeight"].is_number())
        {
            hint.weight = payload["contextWeight"].get<double>();
        }
        if (!std::isfinite(hint.weight))
            hint.weight = 0.35;
        hint.weight = std::max(0.0, std::min(1.0, hint.weight));
        auto trim = [](std::string &s)
        {
            auto start = s.find_first_not_of(" \t\r\n");
            if (start == std::string::npos)
            {
                s.clear();
                return;
            }
            auto end = s.find_last_not_of(" \t\r\n");
            s = s.substr(start, end - start + 1);
        };
        trim(hint.text);
        trim(hint.mode);
        return hint;
    }

    std::string applyContextHintToText(const std::string &text, const ContextHint &hint) const
    {
        if (hint.text.empty() || hint.weight <= 0.0)
            return text;
        std::ostringstream oss;
        oss << "【对话上下文提示 w=" << std::fixed << std::setprecision(2) << hint.weight;
        if (!hint.mode.empty())
            oss << " mode=" << hint.mode;
        oss << "】" << hint.text;
        if (text.empty())
            return oss.str();
        if (hint.weight >= 0.5)
        {
            return oss.str() + "\n" + text;
        }
        return text + "\n" + oss.str();
    }

    std::string applyContextHintToGraph(const std::string &graphContext, const ContextHint &hint) const
    {
        if (hint.text.empty() || hint.weight <= 0.0)
            return graphContext;
        std::ostringstream oss;
        oss << "ctx_stage1";
        if (!hint.mode.empty())
            oss << "|" << hint.mode;
        oss << "|w=" << std::fixed << std::setprecision(2) << hint.weight << ":" << hint.text;
        if (graphContext.empty())
            return oss.str();
        if (hint.weight >= 0.5)
        {
            return oss.str() + "\n" + graphContext;
        }
        return graphContext + "\n" + oss.str();
    }

    static std::vector<float> readFloatArray(const json &arr, size_t maxItems = 8192)
    {
        std::vector<float> out;
        if (!arr.is_array())
            return out;
        out.reserve(std::min(arr.size(), maxItems));
        for (size_t i = 0; i < arr.size() && out.size() < maxItems; i++)
        {
            if (arr[i].is_number())
            {
                out.push_back(arr[i].get<float>());
            }
        }
        return out;
    }

    static std::vector<std::vector<float>> chunkEmbedding(const std::vector<float> &vec, int dim, int maxChunks = 8)
    {
        std::vector<std::vector<float>> out;
        if (vec.empty() || dim <= 0)
            return out;
        int total = (int)vec.size();
        int chunks = std::min(maxChunks, (total + dim - 1) / dim);
        out.reserve((size_t)chunks);
        for (int c = 0; c < chunks; c++)
        {
            std::vector<float> v((size_t)dim, 0.0f);
            int start = c * dim;
            int end = std::min(total, start + dim);
            for (int i = start; i < end; i++)
            {
                v[(size_t)(i - start)] = vec[(size_t)i];
            }
            float norm = 0.0f;
            for (float x : v)
                norm += x * x;
            norm = std::sqrt(std::max(1e-9f, norm));
            for (auto &x : v)
                x /= norm;
            out.push_back(std::move(v));
        }
        return out;
    }

    static std::string buildImageContextFromJson(const json &img)
    {
        if (!img.is_object())
            return "";
        if (img.contains("graphContext") && img["graphContext"].is_string())
        {
            return img["graphContext"].get<std::string>();
        }
        std::ostringstream oss;
        bool has = false;
        if (img.contains("detections") && img["detections"].is_array())
        {
            oss << "vision|labels:";
            size_t count = 0;
            for (auto &d : img["detections"])
            {
                if (!d.is_object())
                    continue;
                std::string label = d.value("label", "");
                double score = d.value("score", 0.0);
                if (label.empty())
                    continue;
                if (count++)
                    oss << ",";
                oss << label << "(" << std::fixed << std::setprecision(2) << score << ")";
                if (count >= 16)
                    break;
            }
            has = count > 0;
        }
        if (img.contains("details") && img["details"].is_array())
        {
            if (has)
                oss << "|";
            oss << "details:";
            size_t count = 0;
            for (auto &d : img["details"])
            {
                if (!d.is_object())
                    continue;
                std::string label = d.value("label", "");
                double score = d.value("score", 0.0);
                if (label.empty())
                    continue;
                if (count++)
                    oss << ",";
                oss << label << "(" << std::fixed << std::setprecision(2) << score << ")";
                if (count >= 16)
                    break;
            }
            has = has || (count > 0);
        }
        if (!has)
            return "";
        return oss.str();
    }

    static std::string extractImageContext(const json &payload)
    {
        if (payload.contains("imageContext") && payload["imageContext"].is_string())
        {
            return payload["imageContext"].get<std::string>();
        }
        if (payload.contains("image") && payload["image"].is_object())
        {
            return buildImageContextFromJson(payload["image"]);
        }
        if (payload.contains("vision") && payload["vision"].is_object())
        {
            return buildImageContextFromJson(payload["vision"]);
        }
        if (payload.contains("imageEmbedding") || payload.contains("imageEmbeddings"))
        {
            return "vision|embedding";
        }
        return "";
    }

    static std::vector<std::vector<float>> extractImageEmbeddings(const json &payload, int dim)
    {
        std::vector<std::vector<float>> out;
        if (payload.contains("imageEmbeddings") && payload["imageEmbeddings"].is_array())
        {
            for (auto &e : payload["imageEmbeddings"])
            {
                auto vec = readFloatArray(e);
                auto chunks = chunkEmbedding(vec, dim, 1);
                if (!chunks.empty())
                    out.push_back(std::move(chunks.front()));
            }
        }
        if (payload.contains("imageEmbedding"))
        {
            auto vec = readFloatArray(payload["imageEmbedding"]);
            auto chunks = chunkEmbedding(vec, dim, 8);
            out.insert(out.end(), chunks.begin(), chunks.end());
        }
        if (payload.contains("image") && payload["image"].is_object())
        {
            auto &img = payload["image"];
            if (img.contains("embeddings") && img["embeddings"].is_array())
            {
                for (auto &e : img["embeddings"])
                {
                    auto vec = readFloatArray(e);
                    auto chunks = chunkEmbedding(vec, dim, 1);
                    if (!chunks.empty())
                        out.push_back(std::move(chunks.front()));
                }
            }
            if (img.contains("embedding"))
            {
                auto vec = readFloatArray(img["embedding"]);
                auto chunks = chunkEmbedding(vec, dim, 8);
                out.insert(out.end(), chunks.begin(), chunks.end());
            }
        }
        if (payload.contains("vision") && payload["vision"].is_object())
        {
            auto &img = payload["vision"];
            if (img.contains("embeddings") && img["embeddings"].is_array())
            {
                for (auto &e : img["embeddings"])
                {
                    auto vec = readFloatArray(e);
                    auto chunks = chunkEmbedding(vec, dim, 1);
                    if (!chunks.empty())
                        out.push_back(std::move(chunks.front()));
                }
            }
            if (img.contains("embedding"))
            {
                auto vec = readFloatArray(img["embedding"]);
                auto chunks = chunkEmbedding(vec, dim, 8);
                out.insert(out.end(), chunks.begin(), chunks.end());
            }
        }
        return out;
    }

    static void injectImageContext(json &payload, const std::string &imageContext)
    {
        if (imageContext.empty())
            return;
        payload["imageContext"] = imageContext;
        if (payload.contains("text") && payload["text"].is_string())
        {
            payload["text"] = payload["text"].get<std::string>() + "\n" + imageContext;
        }
        else if (payload.contains("message") && payload["message"].is_string())
        {
            payload["message"] = payload["message"].get<std::string>() + "\n" + imageContext;
        }
    }

    std::string buildInferenceCacheKey(const std::string &sessionId,
                                       const std::string &text,
                                       const std::string &graphContext,
                                       int maxTokens) const
    {
        std::ostringstream oss;
        oss << "v3|s=" << sessionId << "|t=" << text << "|g=" << graphContext << "|m=" << maxTokens;
        return sha1Hex(oss.str());
    }

    std::vector<std::vector<float>> buildGraphEmbeddings(const json &graphResult, int dim, int rounds = 1)
    {
        auto runtime = activeRuntime();
        if (!runtime)
            return {};
        std::vector<std::string> ids;
        if (graphResult.contains("memes") && graphResult["memes"].is_array())
        {
            for (auto &m : graphResult["memes"])
            {
                if (!m.is_string())
                    continue;
                ids.push_back(m.get<std::string>());
                if (ids.size() >= 64)
                    break;
            }
        }
        if (ids.empty())
            return {};

        auto buildLexicalEmbeddings = [&]() -> std::vector<std::vector<float>>
        {
            std::vector<std::vector<float>> out;
            out.reserve(ids.size());
            for (auto &id : ids)
            {
                auto words = runtime->getMemeWords(id);
                std::ostringstream oss;
                oss << id;
                for (size_t i = 0; i < words.size() && i < 6; i++)
                {
                    oss << ' ' << words[i];
                }
                out.push_back(textToMiniEmbedding(oss.str(), dim));
            }
            return out;
        };

        if (matrixCache_ && matrixCache_->enabled())
        {
            std::ostringstream oss;
            oss << "graphEmb|d=" << dim << "|r=" << rounds << "|";
            for (size_t i = 0; i < ids.size(); i++)
            {
                if (i)
                    oss << ",";
                oss << ids[i];
            }
            std::string key = sha1Hex(oss.str());
            std::vector<std::vector<float>> cached;
            if (matrixCache_->getMatrix(key, (int)ids.size(), dim, cached))
            {
                return cached;
            }
            std::vector<std::vector<float>> out;
            if (auto rt = std::dynamic_pointer_cast<RuntimeState>(runtime))
            {
                GraphTensorBridge bridge(&rt->kvm(), nullptr);
                auto base = bridge.computeNodeEmbeddings(ids, dim);
                out = rt->graph().messagePassingEmbeddings(ids, base, rounds, 0.65f, 0.35f);
            }
            else
            {
                out = buildLexicalEmbeddings();
            }
            matrixCache_->recordAndMaybeSet(key, out, 900, 3);
            return out;
        }
        if (auto rt = std::dynamic_pointer_cast<RuntimeState>(runtime))
        {
            GraphTensorBridge bridge(&rt->kvm(), nullptr);
            auto base = bridge.computeNodeEmbeddings(ids, dim);
            return rt->graph().messagePassingEmbeddings(ids, base, rounds, 0.65f, 0.35f);
        }
        return buildLexicalEmbeddings();
    }

    std::vector<transformer::TrainSample> parseTrainSamples(const json &payload)
    {
        std::vector<transformer::TrainSample> samples;
        if (!payload.contains("samples") || !payload["samples"].is_array())
            return samples;
        for (auto &s : payload["samples"])
        {
            if (!s.is_object())
                continue;
            transformer::TrainSample sample;
            bool cleanedInput = false;
            bool cleanedTarget = false;
            bool cleanedGraph = false;
            sample.input = sanitizeInputText(s.value("input", ""), cleanedInput);
            sample.target = sanitizeInputText(s.value("target", ""), cleanedTarget);
            sample.graph = sanitizeInputText(s.value("graph", ""), cleanedGraph);
            if (cleanedInput || cleanedTarget || cleanedGraph)
                cleanedSampleCount_.fetch_add(1, std::memory_order_relaxed);
            if (sample.input.empty() || sample.target.empty())
                continue;
            samples.push_back(std::move(sample));
        }
        return samples;
    }

    std::vector<std::string> deriveGnnStyleKeywords(const std::string &text, size_t maxKeywords = 8) const
    {
        std::vector<std::string> out;
        auto runtime = activeRuntime();
        if (!runtime)
            return out;

        auto words = tokenize(text);
        if (words.empty())
            return out;

        auto seeds = runtime->mapWordsToMemes(words);
        std::vector<std::pair<std::string, double>> rankedSeeds(seeds.begin(), seeds.end());
        std::sort(rankedSeeds.begin(), rankedSeeds.end(), [](const auto &a, const auto &b)
                  { return a.second > b.second; });

        std::unordered_set<std::string> seen;
        for (const auto &seed : rankedSeeds)
        {
            auto linked = runtime->getMemeWords(seed.first);
            for (const auto &w : linked)
            {
                std::string kw = trimCopy(w);
                if (kw.size() < 2 || kw.size() > 24)
                    continue;
                if (seen.insert(kw).second)
                    out.push_back(kw);
                if (out.size() >= maxKeywords)
                    return out;
            }
        }

        for (const auto &w : words)
        {
            std::string kw = trimCopy(w);
            if (kw.size() < 2 || kw.size() > 24)
                continue;
            if (seen.insert(kw).second)
                out.push_back(kw);
            if (out.size() >= maxKeywords)
                break;
        }
        return out;
    }

    json buildExternalStyleFeatures(const std::string &text, const std::vector<std::string> &keywords) const
    {
        static const std::unordered_set<std::string> positiveLex = {
            "good", "great", "excellent", "clear", "smart", "helpful", "fast", "friendly", "professional", "confident", "precise", "stable", "strong", "clean"};
        static const std::unordered_set<std::string> negativeLex = {
            "bad", "poor", "slow", "confused", "messy", "unclear", "angry", "rude", "broken", "error", "fail", "failure", "chaotic", "weak"};

        std::unordered_set<std::string> normalizedKeywords;
        normalizedKeywords.reserve(keywords.size());
        for (const auto &kwRaw : keywords)
        {
            std::string kw = trimCopy(kwRaw);
            if (kw.empty())
                continue;
            std::transform(kw.begin(), kw.end(), kw.begin(), [](unsigned char c)
                           { return static_cast<char>(std::tolower(c)); });
            normalizedKeywords.insert(kw);
        }

        auto words = tokenize(text);
        size_t keywordHits = 0;
        size_t posHits = 0;
        size_t negHits = 0;
        for (auto w : words)
        {
            std::transform(w.begin(), w.end(), w.begin(), [](unsigned char c)
                           { return static_cast<char>(std::tolower(c)); });
            if (normalizedKeywords.find(w) != normalizedKeywords.end())
                keywordHits++;
            if (positiveLex.find(w) != positiveLex.end())
                posHits++;
            if (negativeLex.find(w) != negativeLex.end())
                negHits++;
        }

        size_t punctuationCount = 0;
        for (unsigned char ch : text)
        {
            if (std::ispunct(ch))
                punctuationCount++;
        }

        const double tokenCount = static_cast<double>(words.size());
        const double keywordDensity = tokenCount > 0.0 ? static_cast<double>(keywordHits) / tokenCount : 0.0;

        const double sentiDen = static_cast<double>(posHits + negHits);
        double sentiment = 0.0;
        if (sentiDen > 0.0)
        {
            sentiment = (static_cast<double>(posHits) - static_cast<double>(negHits)) / sentiDen;
        }

        const double textLen = static_cast<double>(text.size());
        const double punctuationDensity = textLen > 0.0 ? static_cast<double>(punctuationCount) / textLen : 0.0;

        return json{{"keywordDensity", std::clamp(keywordDensity, 0.0, 1.0)},
                    {"sentiment", std::clamp(sentiment, -1.0, 1.0)},
                    {"punctuationDensity", std::clamp(punctuationDensity, 0.0, 1.0)}};
    }

    json buildExternalStyleTrainPayload(bool usingBitNet,
                                        const json &payload,
                                        const std::string &text,
                                        double observedAlign,
                                        const std::vector<std::string> &keywords) const
    {
        json features = buildExternalStyleFeatures(text, keywords);
        double graphEdgeWeight = payload.value("graphEdgeWeight", payload.value("edgeWeight", payload.value("avgEdgeWeight", 0.0)));
        double graphResidual = payload.value("graphResidual", payload.value("residual", payload.value("avgResidual", 0.0)));
        graphEdgeWeight = std::clamp(graphEdgeWeight, 0.0, 1.0);
        graphResidual = std::clamp(graphResidual, 0.0, 1.0);
        return json{{"provider", usingBitNet ? "bitnet" : "llamacpp"},
                    {"model", usingBitNet ? config_.bitnetModel : config_.llamaCppModel},
                    {"text", text},
                    {"style", trimCopy(payload.value("style", payload.value("persona", payload.value("styleProfile", ""))))},
                    {"targetAlign", payload.value("targetAlign", payload.value("styleTargetAlign", 0.85))},
                    {"observedAlign", payload.value("observedAlign", observedAlign)},
                    {"lossScale", payload.value("lossScale", 1.0)},
                    {"styleKeywords", keywords},
                    {"features", features},
                    {"keywordDensity", features.value("keywordDensity", 0.0)},
                    {"sentiment", features.value("sentiment", 0.0)},
                    {"punctuationDensity", features.value("punctuationDensity", 0.0)},
                    {"graphEdgeWeight", graphEdgeWeight},
                    {"graphResidual", graphResidual}};
    }

    std::vector<std::string> parseStyleKeywords(const json &payload, const std::string &text) const
    {
        std::vector<std::string> keywords;
        if (payload.contains("styleKeywords") && payload["styleKeywords"].is_array())
        {
            for (const auto &it : payload["styleKeywords"])
            {
                if (it.is_string())
                {
                    std::string kw = trimCopy(it.get<std::string>());
                    if (!kw.empty())
                        keywords.push_back(kw);
                }
                if (keywords.size() >= 8)
                    break;
            }
        }
        if (keywords.empty())
            keywords = deriveGnnStyleKeywords(text, 8);
        return keywords;
    }

    json postExternalStyleTrainStep(const std::string &provider,
                                    const std::string &baseUrl,
                                    const std::string &model,
                                    int timeoutMs,
                                    const json &payload) const
    {
        json out = json::object();
        out["provider"] = provider;
        out["ok"] = false;
        try
        {
            std::string endpoint = trimCopy(baseUrl);
            if (endpoint.empty())
                endpoint = provider == "bitnet" ? "http://127.0.0.1:8090" : "http://127.0.0.1:8080";
            auto client = drogon::HttpClient::newHttpClient(endpoint);
            if (!client)
            {
                out["error"] = provider + " client init failed";
                return out;
            }

            auto callPath = [&](const std::string &path) -> json
            {
                json ret = json::object();
                auto req = drogon::HttpRequest::newHttpRequest();
                req->setMethod(drogon::Post);
                req->setPath(path);
                req->setContentTypeCode(drogon::CT_APPLICATION_JSON);
                req->setBody(payload.dump());

                std::promise<std::pair<drogon::ReqResult, drogon::HttpResponsePtr>> promise;
                auto future = promise.get_future();
                client->sendRequest(req, [&promise](drogon::ReqResult result, const drogon::HttpResponsePtr &resp)
                                    {
                    try
                    {
                        promise.set_value({result, resp});
                    }
                    catch (...) {}
                });

                if (future.wait_for(std::chrono::milliseconds(std::max(1000, timeoutMs))) != std::future_status::ready)
                {
                    ret["ok"] = false;
                    ret["timeout"] = true;
                    ret["error"] = provider + " style train timeout";
                    return ret;
                }

                auto pair = future.get();
                auto reqResult = pair.first;
                auto resp = pair.second;
                if (reqResult != drogon::ReqResult::Ok || !resp)
                {
                    ret["ok"] = false;
                    ret["error"] = provider + " style train request failed";
                    return ret;
                }
                ret["status"] = (int)resp->statusCode();
                if (resp->statusCode() < 200 || resp->statusCode() >= 300)
                {
                    ret["ok"] = false;
                    ret["error"] = provider + " style train bad status";
                    return ret;
                }

                json doc = json::parse(resp->getBody(), nullptr, false);
                if (doc.is_discarded())
                {
                    ret["ok"] = false;
                    ret["error"] = provider + " style train invalid json";
                    return ret;
                }
                ret["ok"] = true;
                ret["result"] = doc;
                return ret;
            };

            json first = callPath("/api/style/adapter/train_step");
            if (first.value("ok", false))
                return first;

            int firstStatus = first.value("status", 0);
            if (firstStatus == 404 || firstStatus == 405 || firstStatus == 0)
            {
                json second = callPath("/v1/style/adapter/train_step");
                if (second.value("ok", false))
                    return second;
                second["fallback"] = "tried";
                return second;
            }
            return first;
        }
        catch (const std::exception &e)
        {
            out["error"] = e.what();
            return out;
        }
        catch (...)
        {
            out["error"] = provider + " style train unknown error";
            return out;
        }
    }

    void dispatchExternalStyleTrainStep(const std::string &backendMode,
                                        const json &payload,
                                        const std::string &text,
                                        double gnnAlign)
    {
        if (!config_.externalBackendStyleSim)
            return;
        std::string mode = trimCopy(backendMode);
        std::transform(mode.begin(), mode.end(), mode.begin(), [](unsigned char c)
                       { return static_cast<char>(std::tolower(c)); });
        bool usingLlamaCpp = (mode == "llamacpp" || mode == "llama.cpp" || mode == "llama_cpp");
        bool usingBitNet = (mode == "bitnet");
        if (!usingLlamaCpp && !usingBitNet)
            return;

        std::vector<std::string> keywords = parseStyleKeywords(payload, text);
        json trainPayload = buildExternalStyleTrainPayload(usingBitNet, payload, text, gnnAlign, keywords);

        json trainResp = usingBitNet
                             ? postExternalStyleTrainStep("bitnet", config_.bitnetBaseUrl, config_.bitnetModel, config_.bitnetRequestTimeoutMs, trainPayload)
                             : postExternalStyleTrainStep("llamacpp", config_.llamaCppBaseUrl, config_.llamaCppModel, config_.llamaCppRequestTimeoutMs, trainPayload);
        if (!trainResp.value("ok", false) && logger_.enabled())
        {
            logger_.log(LoggerCXX::Type::WARNING,
                        std::string("external style train dispatch failed provider=") + (usingBitNet ? "bitnet" : "llamacpp") +
                            " error=" + trainResp.value("error", std::string("unknown")));
        }
    }

    bool isExternalBackendMode(std::string mode) const
    {
        mode = trimCopy(mode);
        std::transform(mode.begin(), mode.end(), mode.begin(), [](unsigned char c)
                       { return static_cast<char>(std::tolower(c)); });
        return mode == "llamacpp" || mode == "llama.cpp" || mode == "llama_cpp" || mode == "bitnet";
    }

    json bridgeLearnerResultToExternalBackend(const std::string &learner,
                                              const std::string &backendMode,
                                              const json &requestBody,
                                              const json &learnerResult)
    {
        std::string mode = trimCopy(backendMode);
        std::transform(mode.begin(), mode.end(), mode.begin(), [](unsigned char c)
                       { return static_cast<char>(std::tolower(c)); });
        bool usingLlamaCpp = (mode == "llamacpp" || mode == "llama.cpp" || mode == "llama_cpp");
        bool usingBitNet = (mode == "bitnet");
        if (!usingLlamaCpp && !usingBitNet)
        {
            return json{{"applied", false}, {"reason", "non-external-backend"}};
        }

        std::vector<std::pair<std::string, double>> trainItems;
        if (learnerResult.contains("evals") && learnerResult["evals"].is_array())
        {
            for (const auto &it : learnerResult["evals"])
            {
                std::string text = trimCopy(it.value("reply", ""));
                if (text.empty())
                    continue;
                trainItems.push_back({text, std::clamp(it.value("score", 0.0), 0.0, 1.0)});
                if (trainItems.size() >= 24)
                    break;
            }
        }
        if (trainItems.empty() && learnerResult.contains("rounds") && learnerResult["rounds"].is_array())
        {
            for (const auto &round : learnerResult["rounds"])
            {
                auto results = round.value("results", json::array());
                if (!results.is_array())
                    continue;
                for (const auto &item : results)
                {
                    std::string text = trimCopy(item.value("text", ""));
                    if (text.empty())
                        continue;
                    trainItems.push_back({text, std::clamp(item.value("quality", 0.0), 0.0, 1.0)});
                    if (trainItems.size() >= 24)
                        break;
                }
                if (trainItems.size() >= 24)
                    break;
            }
        }
        if (trainItems.empty() && requestBody.contains("samples") && requestBody["samples"].is_array())
        {
            double align = 0.5;
            if (learnerResult.contains("best") && learnerResult["best"].is_object())
                align = std::clamp(learnerResult["best"].value("avgQuality", 0.5), 0.0, 1.0);
            for (const auto &s : requestBody["samples"])
            {
                if (!s.is_string())
                    continue;
                std::string text = trimCopy(s.get<std::string>());
                if (text.empty())
                    continue;
                trainItems.push_back({text, align});
                if (trainItems.size() >= 24)
                    break;
            }
        }

        if (trainItems.empty())
        {
            return json{{"applied", false}, {"reason", "no-train-items"}};
        }

        json stats = json{{"applied", true},
                          {"learner", learner},
                          {"provider", usingBitNet ? "bitnet" : "llamacpp"},
                          {"steps", 0},
                          {"ok", 0},
                          {"failed", 0},
                          {"errors", json::array()}};

        const int maxThreads = std::max(1, config_.inferWorkers);
        const int threadCount = std::max(1, std::min<int>(maxThreads, (int)trainItems.size()));
        stats["threads"] = threadCount;

        std::atomic<size_t> index{0};
        std::atomic<int> stepCount{0};
        std::atomic<int> okCount{0};
        std::atomic<int> failCount{0};
        std::mutex errMu;
        std::vector<std::string> errors;

        auto worker = [&]()
        {
            while (true)
            {
                size_t i = index.fetch_add(1);
                if (i >= trainItems.size())
                    return;

                const std::string &text = trainItems[i].first;
                const double observed = trainItems[i].second;
                json merged = requestBody;
                merged["style"] = learner;
                merged["lossScale"] = requestBody.value("lossScale", 1.0);
                merged["targetAlign"] = requestBody.value("targetAlign", 0.85);
                merged["observedAlign"] = observed;
                merged["learner"] = learner;
                merged["source"] = "learning-bridge";
                if (learnerResult.contains("best") && learnerResult["best"].is_object())
                {
                    merged["graphEdgeWeight"] = learnerResult["best"].value("edgeWeight", 0.0);
                    merged["graphResidual"] = learnerResult["best"].value("avgResidual", 0.0);
                }
                else
                {
                    merged["graphEdgeWeight"] = learnerResult.value("avgEdgeWeight", 0.0);
                    merged["graphResidual"] = learnerResult.value("avgResidual", learnerResult.value("residual", 0.0));
                }

                auto keywords = parseStyleKeywords(merged, text);
                auto trainPayload = buildExternalStyleTrainPayload(usingBitNet, merged, text, observed, keywords);
                trainPayload["learningBridge"] = json{{"learner", learner}, {"backend", mode}, {"groupProc", isGroupProc()}};

                json step = usingBitNet
                                ? postExternalStyleTrainStep("bitnet", config_.bitnetBaseUrl, config_.bitnetModel, config_.bitnetRequestTimeoutMs, trainPayload)
                                : postExternalStyleTrainStep("llamacpp", config_.llamaCppBaseUrl, config_.llamaCppModel, config_.llamaCppRequestTimeoutMs, trainPayload);

                stepCount.fetch_add(1);
                if (step.value("ok", false))
                {
                    okCount.fetch_add(1);
                }
                else
                {
                    failCount.fetch_add(1);
                    std::lock_guard<std::mutex> lock(errMu);
                    errors.push_back(step.value("error", std::string("unknown")));
                }
            }
        };

        std::vector<std::thread> threads;
        threads.reserve((size_t)threadCount);
        for (int i = 0; i < threadCount; i++)
        {
            threads.emplace_back(worker);
        }
        for (auto &t : threads)
        {
            if (t.joinable())
                t.join();
        }

        stats["steps"] = stepCount.load();
        stats["ok"] = okCount.load();
        stats["failed"] = failCount.load();
        for (const auto &e : errors)
            stats["errors"].push_back(e);

        return stats;
    }

    bool parseBoundedInt64(const json &payload, const char *key, int64_t minValue, int64_t maxValue, int64_t &out, std::string &err, bool required = false) const
    {
        if (!payload.contains(key))
        {
            if (required)
            {
                err = std::string(key) + " required";
                return false;
            }
            return true;
        }
        const auto &v = payload[key];
        if (!(v.is_number_integer() || v.is_number_unsigned()))
        {
            err = std::string(key) + " must be integer";
            return false;
        }
        int64_t parsed = 0;
        try
        {
            parsed = v.get<int64_t>();
        }
        catch (...)
        {
            err = std::string(key) + " out of int64 range";
            return false;
        }
        if (parsed < minValue || parsed > maxValue)
        {
            std::ostringstream oss;
            oss << key << " out of range [" << minValue << ", " << maxValue << "]";
            err = oss.str();
            return false;
        }
        out = parsed;
        return true;
    }

    uint64_t currentRssBytes() const
    {
        auto mem = getMemoryUsage();
        if (mem.contains("rss") && mem["rss"].is_number_integer())
        {
            return (uint64_t)std::max<int64_t>(0, mem["rss"].get<int64_t>());
        }
        return 0;
    }

    json trainMonitorToJson(const TrainMonitorState &s) const
    {
        auto toMB = [](uint64_t bytes)
        { return (double)bytes / 1024.0 / 1024.0; };
        int64_t elapsedMs = 0;
        if (s.startedAtMs > 0)
        {
            int64_t endMs = s.running ? nowEpochMs() : (s.finishedAtMs > 0 ? s.finishedAtMs : s.updatedAtMs);
            elapsedMs = std::max<int64_t>(0, endMs - s.startedAtMs);
        }
        return json{{"id", s.id},
                    {"running", s.running},
                    {"ok", s.ok},
                    {"route", s.route},
                    {"stage", s.stage},
                    {"error", s.error},
                    {"startedAtMs", s.startedAtMs},
                    {"updatedAtMs", s.updatedAtMs},
                    {"finishedAtMs", s.finishedAtMs},
                    {"elapsedMs", elapsedMs},
                    {"rssBefore", s.rssBefore},
                    {"rssPeak", s.rssPeak},
                    {"rssAfter", s.rssAfter},
                    {"rssBeforeMB", toMB(s.rssBefore)},
                    {"rssPeakMB", toMB(s.rssPeak)},
                    {"rssAfterMB", toMB(s.rssAfter)},
                    {"docs", s.docs},
                    {"processedDocs", s.processedDocs},
                    {"samples", s.samples},
                    {"chars", s.chars},
                    {"request", s.requestSummary},
                    {"result", s.resultSummary}};
    }

    uint64_t beginTrainMonitor(const std::string &route, const json &requestSummary)
    {
        std::lock_guard<std::mutex> lock(trainMonitorMu_);
        trainMonitorSeq_ += 1;
        trainMonitorActive_ = TrainMonitorState{};
        trainMonitorActive_.id = trainMonitorSeq_;
        trainMonitorActive_.running = true;
        trainMonitorActive_.route = route;
        trainMonitorActive_.stage = "started";
        trainMonitorActive_.startedAtMs = nowEpochMs();
        trainMonitorActive_.updatedAtMs = trainMonitorActive_.startedAtMs;
        trainMonitorActive_.requestSummary = requestSummary;
        trainMonitorActive_.rssBefore = currentRssBytes();
        trainMonitorActive_.rssPeak = trainMonitorActive_.rssBefore;
        return trainMonitorActive_.id;
    }

    void touchTrainMonitor(const std::string &stage,
                           uint64_t docs,
                           uint64_t processedDocs,
                           uint64_t samples,
                           uint64_t chars,
                           const std::string &error = "")
    {
        std::lock_guard<std::mutex> lock(trainMonitorMu_);
        if (!trainMonitorActive_.running)
            return;
        trainMonitorActive_.stage = stage;
        trainMonitorActive_.updatedAtMs = nowEpochMs();
        trainMonitorActive_.docs = docs;
        trainMonitorActive_.processedDocs = processedDocs;
        trainMonitorActive_.samples = samples;
        trainMonitorActive_.chars = chars;
        if (!error.empty())
            trainMonitorActive_.error = error;
        uint64_t rss = currentRssBytes();
        trainMonitorActive_.rssPeak = std::max(trainMonitorActive_.rssPeak, rss);
    }

    void finishTrainMonitor(bool ok, const json &resultSummary, const std::string &error = "")
    {
        std::lock_guard<std::mutex> lock(trainMonitorMu_);
        if (!trainMonitorActive_.running)
            return;
        trainMonitorActive_.running = false;
        trainMonitorActive_.ok = ok;
        trainMonitorActive_.finishedAtMs = nowEpochMs();
        trainMonitorActive_.updatedAtMs = trainMonitorActive_.finishedAtMs;
        trainMonitorActive_.resultSummary = resultSummary;
        if (!error.empty())
            trainMonitorActive_.error = error;
        trainMonitorActive_.rssAfter = currentRssBytes();
        trainMonitorActive_.rssPeak = std::max(trainMonitorActive_.rssPeak, trainMonitorActive_.rssAfter);
        trainMonitorHistory_.push_front(trainMonitorToJson(trainMonitorActive_));
        while (trainMonitorHistory_.size() > 24)
            trainMonitorHistory_.pop_back();
    }

    json getTrainMonitorStats()
    {
        std::lock_guard<std::mutex> lock(trainMonitorMu_);
        json history = json::array();
        for (const auto &item : trainMonitorHistory_)
            history.push_back(item);
        return json{{"active", trainMonitorToJson(trainMonitorActive_)}, {"history", history}};
    }

    void resetTrainMonitor()
    {
        std::lock_guard<std::mutex> lock(trainMonitorMu_);
        trainMonitorActive_ = TrainMonitorState{};
        trainMonitorHistory_.clear();
    }

    void recordRouteMetric(const std::string &route, long long elapsedMs, bool ok)
    {
        std::lock_guard<std::mutex> lock(routeMetricMu_);
        auto &m = routeMetrics_[route];
        m.calls += 1;
        uint64_t ms = (uint64_t)std::max<long long>(0, elapsedMs);
        m.totalMs += ms;
        m.lastMs = ms;
        m.maxMs = std::max(m.maxMs, ms);
        if (!ok)
            m.errors += 1;
    }

    json getMonitoringStats()
    {
        json out;
        out["uptimeSec"] = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - startedAt_).count();
        out["performance"] = json{{"memory", getMemoryUsage()}, {"processCpuPercent", getProcessCpuPercent()}};
        out["cleaning"] = {
            {"enabled", dataCleaningEnabled_},
            {"maxChars", dataCleanMaxChars_},
            {"cleanedInputs", cleanedInputCount_.load(std::memory_order_relaxed)},
            {"cleanedSamples", cleanedSampleCount_.load(std::memory_order_relaxed)}};
        json routes = json::object();
        {
            std::lock_guard<std::mutex> lock(routeMetricMu_);
            for (const auto &kv : routeMetrics_)
            {
                const auto &m = kv.second;
                uint64_t okCalls = m.calls >= m.errors ? (m.calls - m.errors) : 0;
                routes[kv.first] = {
                    {"calls", m.calls},
                    {"errors", m.errors},
                    {"lastMs", m.lastMs},
                    {"maxMs", m.maxMs},
                    {"avgMs", okCalls ? (m.totalMs / okCalls) : 0}};
            }
        }
        out["routes"] = routes;
        out["training"] = getTrainMonitorStats();
        return out;
    }

    void resetMonitoringStats()
    {
        std::lock_guard<std::mutex> lock(routeMetricMu_);
        routeMetrics_.clear();
        cleanedInputCount_.store(0, std::memory_order_relaxed);
        cleanedSampleCount_.store(0, std::memory_order_relaxed);
        resetTrainMonitor();
    }

    json getRuntimeFeatureState()
    {
        json out;
        out["memebarrier"] = {
            {"enabled", barrier_ ? barrier_->running() : false},
            {"available", !config_.disableBarrier},
            {"threshold", barrier_ ? barrier_->threshold() : MODEL_DEFAULTS.maliciousThreshold},
            {"disableRate", barrier_ ? barrier_->disableRateTarget() : 0.01},
            {"useTextCNN", barrier_ ? barrier_->useTextCNN() : true},
            {"useRecommender", barrier_ ? barrier_->useRecommender() : true},
            {"useTorchModels", barrier_ ? barrier_->useTorchModels() : false}};
        out["learning"] = {
            {"enabled", !(rlDisabled_ && advDisabled_ && gnnGaDisabled_)},
            {"cliDisabled", config_.disableLearning}};
        out["rl"] = {{"enabled", !rlDisabled_}, {"cliDisabled", (config_.disableLearning || config_.disableRL)}};
        out["adv"] = {{"enabled", !advDisabled_}, {"cliDisabled", (config_.disableLearning || config_.disableADV)}};
        out["gnnGa"] = {{"enabled", !gnnGaDisabled_}, {"cliDisabled", config_.disableLearning}};
        out["dialogLearning"] = {{"enabled", dialogLearningEnabled_}, {"transformerEnabled", dialogTransformerLearning_}};
        out["dialogThresholds"] = {{"rlEvery", dialogThresholds_.rlEvery}, {"advEvery", dialogThresholds_.advEvery}, {"gnnEvery", dialogThresholds_.gnnEvery}};
        out["dialogCounters"] = {{"total", dialogCounters_.total}, {"lastRL", dialogCounters_.lastRL}, {"lastADV", dialogCounters_.lastADV}, {"lastGNN", dialogCounters_.lastGNN}};
        out["dataCleaning"] = {{"enabled", dataCleaningEnabled_}, {"maxChars", dataCleanMaxChars_}};
        std::string transformerMode = "off";
        if (config_.inferenceEnabled)
            transformerMode = trimCopy(config_.transformerBackend.empty() ? (config_.usingOllama ? (config_.ollamaFineTuning ? "ollama-fine-tuning" : "ollama") : "native") : config_.transformerBackend);
        out["pipeline"] = {{"gnnModuleDisabled", config_.disableGnnModule},
                           {"contextModuleDisabled", config_.disableContextModule},
                           {"inferenceEnabled", config_.inferenceEnabled},
                           {"usingOllama", config_.usingOllama},
                           {"ollamaFineTuning", config_.ollamaFineTuning},
                           {"transformerMode", transformerMode},
                           {"ollamaModel", config_.ollamaModel},
                           {"llamacppModel", config_.llamaCppModel},
                           {"bitnetModel", config_.bitnetModel},
                           {"shmProtocolVersion", kSharedMemoryProtocolVersion},
                           {"outsidesLinkMode", currentOutsidesLinkMode()}};
        return out;
    }

    json applyRuntimeFeaturePatch(const json &patch)
    {
        json out{{"applied", json::object()}, {"warnings", json::array()}};
        if (patch.contains("memebarrierEnabled") && patch["memebarrierEnabled"].is_boolean())
        {
            bool on = patch["memebarrierEnabled"].get<bool>();
            if (on)
            {
                auto rt = std::dynamic_pointer_cast<RuntimeState>(pool_->getActive()->runtime());
                if (!barrier_ && rt)
                    barrier_ = std::make_shared<MemeBarrier>(rt, 10000, MODEL_DEFAULTS.maliciousThreshold, 5);
                if (barrier_)
                    barrier_->start();
                else
                    out["warnings"].push_back("memebarrier requires RuntimeState; current backend does not expose it");
            }
            else if (barrier_)
            {
                barrier_->stop();
            }
            out["applied"]["memebarrierEnabled"] = on;
        }
        if (patch.contains("maliciousThreshold") && patch["maliciousThreshold"].is_number())
        {
            double v = patch["maliciousThreshold"].get<double>();
            auto rt = std::dynamic_pointer_cast<RuntimeState>(pool_->getActive()->runtime());
            if (!barrier_ && rt)
                barrier_ = std::make_shared<MemeBarrier>(rt, 10000, v, 5);
            if (barrier_)
                barrier_->setThreshold(v);
            out["applied"]["maliciousThreshold"] = v;
        }
        if (patch.contains("memebarrierDisableRate") && patch["memebarrierDisableRate"].is_number())
        {
            double v = patch["memebarrierDisableRate"].get<double>();
            auto rt = std::dynamic_pointer_cast<RuntimeState>(pool_->getActive()->runtime());
            if (!barrier_ && rt)
                barrier_ = std::make_shared<MemeBarrier>(rt, 10000, MODEL_DEFAULTS.maliciousThreshold, 5);
            if (barrier_)
                barrier_->setDisableRateTarget(v);
            out["applied"]["memebarrierDisableRate"] = v;
        }
        if (patch.contains("memebarrierUseCNN") && patch["memebarrierUseCNN"].is_boolean())
        {
            bool on = patch["memebarrierUseCNN"].get<bool>();
            auto rt = std::dynamic_pointer_cast<RuntimeState>(pool_->getActive()->runtime());
            if (!barrier_ && rt)
                barrier_ = std::make_shared<MemeBarrier>(rt, 10000, MODEL_DEFAULTS.maliciousThreshold, 5);
            if (barrier_)
                barrier_->setUseTextCNN(on);
            out["applied"]["memebarrierUseCNN"] = on;
        }
        if (patch.contains("memebarrierUseRecommender") && patch["memebarrierUseRecommender"].is_boolean())
        {
            bool on = patch["memebarrierUseRecommender"].get<bool>();
            auto rt = std::dynamic_pointer_cast<RuntimeState>(pool_->getActive()->runtime());
            if (!barrier_ && rt)
                barrier_ = std::make_shared<MemeBarrier>(rt, 10000, MODEL_DEFAULTS.maliciousThreshold, 5);
            if (barrier_)
                barrier_->setUseRecommender(on);
            out["applied"]["memebarrierUseRecommender"] = on;
        }
        if (patch.contains("memebarrierUseTorch") && patch["memebarrierUseTorch"].is_boolean())
        {
            bool on = patch["memebarrierUseTorch"].get<bool>();
            auto rt = std::dynamic_pointer_cast<RuntimeState>(pool_->getActive()->runtime());
            if (!barrier_ && rt)
                barrier_ = std::make_shared<MemeBarrier>(rt, 10000, MODEL_DEFAULTS.maliciousThreshold, 5);
            if (barrier_)
                barrier_->setUseTorchModels(on);
            out["applied"]["memebarrierUseTorch"] = on;
        }
        if (patch.contains("learningEnabled") && patch["learningEnabled"].is_boolean())
        {
            bool on = patch["learningEnabled"].get<bool>();
            if (!on)
            {
                rlDisabled_ = true;
                advDisabled_ = true;
                gnnGaDisabled_ = true;
                dialogLearningEnabled_ = false;
            }
            else
            {
                if (config_.disableLearning)
                {
                    out["warnings"].push_back("learning was CLI-disabled; runtime override enabled");
                }
                else
                {
                    gnnGaDisabled_ = false;
                }
            }
            out["applied"]["learningEnabled"] = on;
        }
        if (patch.contains("rlEnabled") && patch["rlEnabled"].is_boolean())
        {
            bool on = patch["rlEnabled"].get<bool>();
            if (on && (config_.disableLearning || config_.disableRL))
            {
                out["warnings"].push_back("rl was CLI-disabled; runtime override enabled");
            }
            rlDisabled_ = !on;
            out["applied"]["rlEnabled"] = on;
        }
        if (patch.contains("advEnabled") && patch["advEnabled"].is_boolean())
        {
            bool on = patch["advEnabled"].get<bool>();
            if (on && (config_.disableLearning || config_.disableADV))
            {
                out["warnings"].push_back("adv was CLI-disabled; runtime override enabled");
            }
            advDisabled_ = !on;
            out["applied"]["advEnabled"] = on;
        }
        if (patch.contains("gnnGaEnabled") && patch["gnnGaEnabled"].is_boolean())
        {
            bool on = patch["gnnGaEnabled"].get<bool>();
            if (on && config_.disableLearning)
            {
                out["warnings"].push_back("gnnGa was CLI-disabled; runtime override enabled");
            }
            gnnGaDisabled_ = !on;
            out["applied"]["gnnGaEnabled"] = on;
        }
        if (patch.contains("dialogLearningEnabled") && patch["dialogLearningEnabled"].is_boolean())
        {
            dialogLearningEnabled_ = patch["dialogLearningEnabled"].get<bool>();
            out["applied"]["dialogLearningEnabled"] = dialogLearningEnabled_;
        }
        if (patch.contains("dialogTransformerLearningEnabled") && patch["dialogTransformerLearningEnabled"].is_boolean())
        {
            dialogTransformerLearning_ = patch["dialogTransformerLearningEnabled"].get<bool>();
            out["applied"]["dialogTransformerLearningEnabled"] = dialogTransformerLearning_;
        }
        if (patch.contains("rlEvery") && patch["rlEvery"].is_number())
        {
            double v = patch["rlEvery"].get<double>();
            if (std::isfinite(v) && v > 0)
            {
                dialogThresholds_.rlEvery = v;
                out["applied"]["rlEvery"] = dialogThresholds_.rlEvery;
            }
        }
        if (patch.contains("advEvery") && patch["advEvery"].is_number())
        {
            double v = patch["advEvery"].get<double>();
            if (std::isfinite(v) && v > 0)
            {
                dialogThresholds_.advEvery = v;
                out["applied"]["advEvery"] = dialogThresholds_.advEvery;
            }
        }
        if (patch.contains("gnnEvery") && patch["gnnEvery"].is_number())
        {
            double v = patch["gnnEvery"].get<double>();
            if (std::isfinite(v) && v > 0)
            {
                dialogThresholds_.gnnEvery = v;
                out["applied"]["gnnEvery"] = dialogThresholds_.gnnEvery;
            }
        }
        if (patch.contains("dataCleaningEnabled") && patch["dataCleaningEnabled"].is_boolean())
        {
            dataCleaningEnabled_ = patch["dataCleaningEnabled"].get<bool>();
            out["applied"]["dataCleaningEnabled"] = dataCleaningEnabled_;
        }
        if (patch.contains("dataCleanMaxChars") && patch["dataCleanMaxChars"].is_number_integer())
        {
            int v = patch["dataCleanMaxChars"].get<int>();
            if (v >= 128 && v <= 65536)
            {
                dataCleanMaxChars_ = (size_t)v;
                out["applied"]["dataCleanMaxChars"] = dataCleanMaxChars_;
            }
        }
        if (patch.contains("contextModuleDisabled") && patch["contextModuleDisabled"].is_boolean())
        {
            config_.disableContextModule = patch["contextModuleDisabled"].get<bool>();
            out["applied"]["contextModuleDisabled"] = config_.disableContextModule;
        }
        if (patch.contains("contextLayerEnabled") && patch["contextLayerEnabled"].is_boolean())
        {
            bool on = patch["contextLayerEnabled"].get<bool>();
            config_.disableContextModule = !on;
            out["applied"]["contextLayerEnabled"] = on;
            out["applied"]["contextModuleDisabled"] = config_.disableContextModule;
        }
        if (patch.contains("gnnModuleDisabled") && patch["gnnModuleDisabled"].is_boolean())
        {
            config_.disableGnnModule = patch["gnnModuleDisabled"].get<bool>();
            out["applied"]["gnnModuleDisabled"] = config_.disableGnnModule;
        }
        if (patch.contains("usingOllama") && patch["usingOllama"].is_boolean())
        {
            config_.usingOllama = patch["usingOllama"].get<bool>();
            config_.inferenceEnabled = true;
            config_.transformerBackend = config_.usingOllama ? (config_.ollamaFineTuning ? "ollama-fine-tuning" : "ollama") : "native";
            out["applied"]["usingOllama"] = config_.usingOllama;
            out["applied"]["inferenceEnabled"] = config_.inferenceEnabled;
            out["applied"]["transformerMode"] = config_.transformerBackend;
        }
        if (patch.contains("ollamaFineTuning") && patch["ollamaFineTuning"].is_boolean())
        {
            config_.ollamaFineTuning = patch["ollamaFineTuning"].get<bool>();
            if (config_.usingOllama)
                config_.transformerBackend = config_.ollamaFineTuning ? "ollama-fine-tuning" : "ollama";
            out["applied"]["ollamaFineTuning"] = config_.ollamaFineTuning;
            out["applied"]["transformerMode"] = config_.transformerBackend;
        }
        if (patch.contains("inferenceEnabled") && patch["inferenceEnabled"].is_boolean())
        {
            config_.inferenceEnabled = patch["inferenceEnabled"].get<bool>();
            out["applied"]["inferenceEnabled"] = config_.inferenceEnabled;
        }
        if (patch.contains("transformerMode") && patch["transformerMode"].is_string())
        {
            std::string mode = trimCopy(patch["transformerMode"].get<std::string>());
            std::transform(mode.begin(), mode.end(), mode.begin(), [](unsigned char ch)
                           { return static_cast<char>(std::tolower(ch)); });
            if (mode == "native")
            {
                config_.inferenceEnabled = true;
                config_.usingOllama = false;
                config_.ollamaFineTuning = false;
                config_.transformerBackend = "native";
                out["applied"]["transformerMode"] = "native";
            }
            else if (mode == "ollama")
            {
                config_.inferenceEnabled = true;
                config_.usingOllama = true;
                config_.ollamaFineTuning = false;
                config_.transformerBackend = "ollama";
                out["applied"]["transformerMode"] = "ollama";
            }
            else if (mode == "ollama-fine-tuning" || mode == "ollama_fine_tuning")
            {
                config_.inferenceEnabled = true;
                config_.usingOllama = true;
                config_.ollamaFineTuning = true;
                config_.transformerBackend = "ollama-fine-tuning";
                out["applied"]["transformerMode"] = "ollama-fine-tuning";
            }
            else if (mode == "llamacpp" || mode == "llama.cpp" || mode == "llama_cpp")
            {
                config_.inferenceEnabled = true;
                config_.usingOllama = false;
                config_.ollamaFineTuning = false;
                config_.transformerBackend = "llamacpp";
                out["applied"]["transformerMode"] = "llamacpp";
            }
            else if (mode == "bitnet")
            {
                config_.inferenceEnabled = true;
                config_.usingOllama = false;
                config_.ollamaFineTuning = false;
                config_.transformerBackend = "bitnet";
                out["applied"]["transformerMode"] = "bitnet";
            }
            else if (mode == "off" || mode == "false" || mode == "disabled")
            {
                config_.inferenceEnabled = false;
                config_.transformerBackend = "off";
                out["applied"]["transformerMode"] = "off";
            }
            else
            {
                out["warnings"].push_back(std::string("unknown transformerMode: ") + mode);
            }
            out["applied"]["inferenceEnabled"] = config_.inferenceEnabled;
            out["applied"]["usingOllama"] = config_.usingOllama;
            out["applied"]["ollamaFineTuning"] = config_.ollamaFineTuning;
            out["applied"]["transformerBackend"] = config_.transformerBackend;
        }
        return out;
    }

    std::shared_ptr<IReinforcementLearner> ensureRL()
    {
        if (!rl_)
        {
            auto factory = module_mount::ModuleRegistry::instance().reinforcementLearnerFactory();
            if (factory)
                rl_ = factory(pool_, getProjectRoot() / "tests");
            else
                rl_ = std::make_shared<ReinforcementLearner>(pool_, getProjectRoot() / "tests");
        }
        return rl_;
    }

    std::shared_ptr<IAdversarialLearner> ensureADV()
    {
        if (!adv_)
        {
            auto factory = module_mount::ModuleRegistry::instance().adversarialLearnerFactory();
            if (factory)
                adv_ = factory(pool_);
            else
                adv_ = std::make_shared<AdversarialLearner>(pool_);
        }
        return adv_;
    }

    std::shared_ptr<IGnnGaLearner> ensureGnnGa()
    {
        if (!gnnGa_)
        {
            auto factory = module_mount::ModuleRegistry::instance().gnnGaLearnerFactory();
            if (factory)
                gnnGa_ = factory(pool_, getProjectRoot() / "tests");
            else
                gnnGa_ = std::make_shared<GnnGaLearner>(pool_, getProjectRoot() / "tests");
        }
        return gnnGa_;
    }

    std::string buildDialogContextLog(const json &result, const json &payload) const
    {
        std::string text = payload.value("text", payload.value("message", ""));
        std::string reply = result.value("reply", "");
        std::string sessionId = result.value("sessionId", payload.value("sessionId", ""));
        double align = result.value("gnnAlign", 0.0);
        double ctxW = result.value("contextWeight", 0.0);
        std::ostringstream oss;
        oss << "session=" << sessionId << " | align=" << std::fixed << std::setprecision(2) << align
            << " | ctxW=" << std::fixed << std::setprecision(2) << ctxW
            << " | text=" << text << " | reply=" << reply;
        return oss.str();
    }

    std::string buildComputeRouteLog(const json &result) const
    {
        std::ostringstream oss;
        oss << "seeds=";
        if (result.contains("seeds") && result["seeds"].is_array())
        {
            int count = 0;
            for (auto &s : result["seeds"])
            {
                if (s.is_array() && s.size() >= 2 && s[0].is_string())
                {
                    if (count++)
                        oss << ",";
                    oss << s[0].get<std::string>();
                }
                if (count >= 10)
                    break;
            }
        }
        oss << " | memes=";
        if (result.contains("memes") && result["memes"].is_array())
        {
            int count = 0;
            for (auto &m : result["memes"])
            {
                if (!m.is_string())
                    continue;
                if (count++)
                    oss << ",";
                oss << m.get<std::string>();
                if (count >= 10)
                    break;
            }
        }
        if (result.contains("gnnStage2") && result["gnnStage2"].is_string())
        {
            oss << " | stage2=" << result["gnnStage2"].get<std::string>();
        }
        return oss.str();
    }

    std::string buildMemorySnapshotLog() const
    {
        auto ctrl = pool_->getActive();
        if (!ctrl)
            return "snapshot=unavailable";
        auto snap = ctrl->snapshot();
        return std::string("snapshot=") + snap.dump();
    }

    std::string buildMonitoringLog() const
    {
        if (!barrier_)
            return "memebarrier=disabled";
        auto st = barrier_->getStats();
        std::ostringstream oss;
        oss << "memebarrier scans=" << st.scans << " isolated=" << st.isolated
            << " avgScore=" << std::fixed << std::setprecision(3) << st.avgScore
            << " maxScore=" << std::fixed << std::setprecision(3) << st.maxScore;
        return oss.str();
    }

    void logSecurityEvent(const std::string &message) const
    {
        if (logger_.enabled())
            logger_.log(LoggerCXX::Type::SECURITY, message);
    }

    void logRouteError(const std::string &route, const std::string &message) const
    {
        if (logger_.enabled())
            logger_.log(LoggerCXX::Type::ERROR, std::string("route=") + route + " error=" + message);
    }

    void logRouteCompute(const std::string &route, const std::string &message) const
    {
        if (logger_.enabled())
            logger_.log(LoggerCXX::Type::COMPUTE, std::string("route=") + route + " " + message);
    }

    void setupRoutes()
    {
        if (config_.httpLog)
        {
            drogon::app().registerPostHandlingAdvice([](const drogon::HttpRequestPtr &req, const drogon::HttpResponsePtr &resp)
                                                     {
                try {
                    int status = resp ? (int)resp->statusCode() : 0;
                    std::string path = req ? req->path() : std::string();
                    std::string method = req ? req->methodString() : std::string("UNKNOWN");
                    std::cout << "[gateway] " << method << " " << path << " -> " << status << std::endl;
                } catch (...) {
                } });
        }

        drogon::app().registerHandler("/", [](const drogon::HttpRequestPtr &, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            auto resp = drogon::HttpResponse::newHttpResponse();
            resp->setStatusCode(drogon::k404NotFound);
            resp->setBody("UI moved to auth_frontend_server.cjs (default :5081)");
            cb(resp); });

        drogon::app().registerHandler("/api/graph/chat", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            try {
                auto jsonStringOrEmpty = [](const json &obj, const char *key) -> std::string {
                    if (!obj.is_object() || !obj.contains(key) || !obj[key].is_string()) {
                        return std::string();
                    }
                    return obj[key].get<std::string>();
                };
                bool ok = true;
                json body = parseRequestBody(req, ok);
                if (!ok) return badRequest(cb, "invalid json");
                auto controller = pool_->getActive();
                json graphBody = body;
                graphBody["disableAddon"] = true;
                json result = controller->respond(graphBody);
                logRouteCompute("/api/graph/chat", "ok");
                respondJson(cb, json{{"ok", true}, {"result", result}});
                onDialogCompleted(result, body);
            } catch (const std::exception &e) {
                logRouteError("/api/graph/chat", e.what());
                respondJson(cb, json{{"ok", false}, {"error", e.what()}}, drogon::k500InternalServerError);
            } catch (...) {
                logRouteError("/api/graph/chat", "internal error");
                respondJson(cb, json{{"ok", false}, {"error", "internal error"}}, drogon::k500InternalServerError);
            } }, {drogon::Post});

        drogon::app().registerHandler("/api/chat", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            auto routeStarted = std::chrono::steady_clock::now();
            if (config_.httpLog) std::cout << "[gateway] ENTER /api/chat" << std::endl;
            auto finalizeMetric = [&](bool ok) {
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - routeStarted).count();
                recordRouteMetric("/api/chat", elapsed, ok);
            };
            if (!authOK(req)) {
                finalizeMetric(false);
                return unauthorized(cb);
            }
            try {
                auto jsonStringOrEmpty = [](const json &obj, const char *key) -> std::string {
                    if (!obj.is_object() || !obj.contains(key) || !obj[key].is_string()) {
                        return std::string();
                    }
                    return obj[key].get<std::string>();
                };
                bool ok = true;
                json body = parseRequestBody(req, ok);
                if (!ok) {
                    finalizeMetric(false);
                    return badRequest(cb, "invalid json");
                }
                std::string text = body.value("text", body.value("message", ""));
                bool cleanedText = false;
                text = sanitizeInputText(text, cleanedText);
                if (cleanedText)
                    cleanedInputCount_.fetch_add(1, std::memory_order_relaxed);
                if (text.empty()) {
                    finalizeMetric(false);
                    return badRequest(cb, "text required");
                }
                bool simpleQuery = false;
                int tokenCount = roughTokenCount(text);
                bool longQuerySafeMode = tokenCount >= 24 || text.size() >= 360;
                if (!config_.inferenceEnabled) {
                    finalizeMetric(false);
                    return respondJson(cb, json{{"ok", false}, {"error", "inference disabled"}}, drogon::k503ServiceUnavailable);
                }
                std::string backendMode = trimCopy(config_.transformerBackend);
                std::transform(backendMode.begin(), backendMode.end(), backendMode.begin(), [](unsigned char ch)
                               { return static_cast<char>(std::tolower(ch)); });
                if (backendMode.empty())
                    backendMode = config_.usingOllama ? (config_.ollamaFineTuning ? "ollama-fine-tuning" : "ollama") : "native";
                bool usingOllama = (backendMode == "ollama" || backendMode == "ollama-fine-tuning");
                bool usingLlamaCpp = (backendMode == "llamacpp" || backendMode == "llama.cpp" || backendMode == "llama_cpp");
                bool usingBitNet = (backendMode == "bitnet");
                bool usingNativeTransformer = !(usingOllama || usingLlamaCpp || usingBitNet);
                bool gnnModuleDisabled = config_.disableGnnModule;
                bool contextModuleDisabled = config_.disableContextModule;
                bool useGraphSelector = body.value("enableGraphSelector", false);
                bool hardDisableGraphSelector = boolFrom(getEnv("AI_DISABLE_GRAPH_SELECTOR", "true"), true);
                if (hardDisableGraphSelector)
                    useGraphSelector = false;
                if (gnnModuleDisabled)
                    useGraphSelector = false;
                if (usingNativeTransformer)
                {
                    std::lock_guard<std::mutex> lock(transformerMu_);
                    auto tp = transformer_.params();
                    if (tp.value("enableDynamicReasoning", true)) {
                        int th = std::max(2, tp.value("simpleTokenThreshold", 10));
                        simpleQuery = tokenCount > 0 && tokenCount <= th;
                    }
                }
                std::string imageContext = extractImageContext(body);
                if (contextModuleDisabled)
                    imageContext.clear();
                json graphResult = json::object();
                if (useGraphSelector && !longQuerySafeMode) {
                    auto graphStart = std::chrono::steady_clock::now();
                    if (config_.httpLog)
                        std::cout << "[gateway] /api/chat stage=graph_selector_start" << std::endl;
                    json bodyForGraph = body;
                    injectImageContext(bodyForGraph, imageContext);
                    bodyForGraph["disableAddon"] = true;
                    auto controller = pool_->getActive();
                    graphResult = controller->respond(bodyForGraph);
                    if (config_.httpLog)
                    {
                        auto graphMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - graphStart).count();
                        std::cout << "[gateway] /api/chat stage=graph_selector_done elapsedMs=" << graphMs << std::endl;
                    }
                }
                std::string graphContext;
                if (!imageContext.empty()) {
                    if (graphContext.empty()) graphContext = imageContext;
                    else graphContext = graphContext + "\n" + imageContext;
                }
                std::string graphReplyText = graphResult.is_object() ? graphResult.value("reply", "") : std::string();
                if (!graphReplyText.empty()) {
                    if (graphContext.empty()) graphContext = graphReplyText;
                    else graphContext = graphContext + "\n" + graphReplyText;
                }
                int maxTokens = graphResult.value("maxTokens", 128);
                if (body.contains("maxTokens") && body["maxTokens"].is_number_integer()) maxTokens = body["maxTokens"].get<int>();
                if (longQuerySafeMode) maxTokens = std::min(maxTokens, 32);
                maxTokens = std::max(16, maxTokens);
                auto hint = extractContextHint(body);
                if (contextModuleDisabled)
                    hint = ContextHint{};
                auto gnnHint = buildGnnHintSummary(graphResult, body, hint);
                hint = harmonizeHintWithGnn(hint, gnnHint.align);
                if (gnnModuleDisabled)
                    gnnHint = GnnHintSummary{};
                std::string sparkAnn;
                (void)sparkAnn;
                if (!sparkAnn.empty()) {
                    if (graphContext.empty()) graphContext = sparkAnn;
                    else graphContext = graphContext + "\n" + sparkAnn;
                }
                if (!contextModuleDisabled)
                {
                    text = applyContextHintToText(text, hint);
                    graphContext = applyContextHintToGraph(graphContext, hint);
                }
                else
                {
                    graphContext.clear();
                }
                bool cleanedGraph = false;
                graphContext = sanitizeInputText(graphContext, cleanedGraph);
                if (cleanedGraph)
                    cleanedInputCount_.fetch_add(1, std::memory_order_relaxed);
                bool addonContractEnabled = body.value("enableAddonToolContract", body.value("enableAddonTools", false));
                if (longQuerySafeMode) addonContractEnabled = false;
                if (!simpleQuery && addonContractEnabled) {
                    graphContext = attachAddonToolContract(graphContext);
                }
                std::string sessionId = body.value("sessionId", "");
                std::string providerTag = std::string("provider:") + backendMode;
                std::string cacheKey = buildInferenceCacheKey(sessionId, text, providerTag + "\n" + graphContext, maxTokens);
                json tReply;
                json verify;
                bool cacheHit = false;
                long long modelElapsedMs = 0;
                long long verifyElapsedMs = 0;
                int imageEmbCount = 0;
                if (db_) {
                    json cached;
                    if (db_->getInferenceCache(cacheKey, cached)) {
                        cacheHit = true;
                        tReply = cached.contains("tReply") ? cached["tReply"] : cached;
                        if (cached.contains("verify")) verify = cached["verify"];
                    }
                }
                if (cacheHit && verify.is_null()) {
                    verify = json{{"ok", true}, {"cached", true}};
                }
                if (!cacheHit) {
                    if (usingLlamaCpp || usingBitNet)
                        dispatchExternalStyleTrainStep(backendMode, body, text, gnnHint.align);
                    std::string modelGraphContext = graphContext;
                    if (usingOllama)
                    {
                        auto t0 = std::chrono::steady_clock::now();
                        if (config_.httpLog)
                            std::cout << "[gateway] /api/chat stage=model_start provider=ollama maxTokens=" << maxTokens << std::endl;
                        tReply = chatWithOllama(text, modelGraphContext, maxTokens);
                        modelElapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
                        if (config_.httpLog)
                            std::cout << "[gateway] /api/chat stage=model_done provider=ollama elapsedMs=" << modelElapsedMs << std::endl;
                    }
                    else if (usingLlamaCpp)
                    {
                        auto t0 = std::chrono::steady_clock::now();
                        if (config_.httpLog)
                            std::cout << "[gateway] /api/chat stage=model_start provider=llamacpp maxTokens=" << maxTokens << std::endl;
                        tReply = chatWithLlamaCpp(text, modelGraphContext, maxTokens);
                        modelElapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
                        if (config_.httpLog)
                            std::cout << "[gateway] /api/chat stage=model_done provider=llamacpp elapsedMs=" << modelElapsedMs << std::endl;
                    }
                    else if (usingBitNet)
                    {
                        auto t0 = std::chrono::steady_clock::now();
                        if (config_.httpLog)
                            std::cout << "[gateway] /api/chat stage=model_start provider=bitnet maxTokens=" << maxTokens << std::endl;
                        tReply = chatWithBitNet(text, modelGraphContext, maxTokens);
                        modelElapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
                        if (config_.httpLog)
                            std::cout << "[gateway] /api/chat stage=model_done provider=bitnet elapsedMs=" << modelElapsedMs << std::endl;
                    }
                    else
                    {
                        auto t0 = std::chrono::steady_clock::now();
                        if (config_.httpLog)
                            std::cout << "[gateway] /api/chat stage=model_start provider=transformer maxTokens=" << maxTokens << std::endl;
                        std::lock_guard<std::mutex> lock(transformerMu_);
                        int dim = transformer_.params().value("dModel", 128);
                        int rounds = gnnHint.align >= 0.45 ? 2 : 1;
                        std::vector<std::vector<float>> graphEmb;
                        bool enableGraphEmbeddings = body.value("enableGraphEmbeddings", false);
                        if (enableGraphEmbeddings) {
                            graphEmb = buildGraphEmbeddings(graphResult, dim, rounds);
                        }
                        auto imageEmb = extractImageEmbeddings(body, dim);
                        if (!imageEmb.empty()) {
                            imageEmbCount = (int)imageEmb.size();
                            graphEmb.insert(graphEmb.end(), imageEmb.begin(), imageEmb.end());
                        }
                        tReply = transformer_.chat(text, modelGraphContext, graphEmb, maxTokens);
                        modelElapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
                        if (config_.httpLog)
                            std::cout << "[gateway] /api/chat stage=model_done provider=transformer elapsedMs=" << modelElapsedMs << std::endl;
                    }
                    std::string replyCandidate = tReply.value("reply", "");
                    if (longQuerySafeMode) {
                        verify = json{{"ok", false}, {"skipped", true}, {"reason", "long-query-safe-mode"}};
                    } else if (!usingNativeTransformer) {
                        verify = json{{"ok", false}, {"skipped", true}, {"reason", std::string("using-") + backendMode}};
                    } else {
                        auto v0 = std::chrono::steady_clock::now();
                        if (config_.httpLog)
                            std::cout << "[gateway] /api/chat stage=verify_start" << std::endl;
                        std::lock_guard<std::mutex> lock(transformerMu_);
                        verify = transformer_.verify(text, graphContext, replyCandidate);
                        verifyElapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - v0).count();
                        if (config_.httpLog)
                            std::cout << "[gateway] /api/chat stage=verify_done elapsedMs=" << verifyElapsedMs << std::endl;
                    }
                    if (db_) {
                        db_->setInferenceCache(cacheKey, json{{"tReply", tReply}, {"verify", verify}}, 86400);
                    }
                }
                std::string transformerReplyRaw = tReply.value("reply", "");
                std::string reply = stripAddonToolDirective(transformerReplyRaw);
                addon::AddonResult addonExec;
                AddonToolCall addonCall;
                bool addonUsed = false;
                if (addonContractEnabled) {
                    addonCall = parseAddonToolCall(transformerReplyRaw, text);
                    addonUsed = invokeAddonTool(addonCall, body, addonExec);
                }
                if (addonUsed) {
                    std::string followupGraphContext = buildAddonResultContext(graphContext, addonCall, addonExec);
                    if (usingLlamaCpp || usingBitNet)
                        dispatchExternalStyleTrainStep(backendMode, body, text, gnnHint.align);
                    json tReplyAfterTool;
                    if (usingOllama)
                    {
                        tReplyAfterTool = chatWithOllama(text, followupGraphContext, maxTokens);
                    }
                    else if (usingLlamaCpp)
                    {
                        tReplyAfterTool = chatWithLlamaCpp(text, followupGraphContext, maxTokens);
                    }
                    else if (usingBitNet)
                    {
                        tReplyAfterTool = chatWithBitNet(text, followupGraphContext, maxTokens);
                    }
                    else
                    {
                        std::lock_guard<std::mutex> lock(transformerMu_);
                        tReplyAfterTool = transformer_.chat(text, followupGraphContext, maxTokens);
                    }
                    std::string rewritten = stripAddonToolDirective(tReplyAfterTool.value("reply", ""));
                    if (!rewritten.empty()) {
                        reply = rewritten;
                        transformerReplyRaw = tReplyAfterTool.value("reply", transformerReplyRaw);
                    } else {
                        reply = addonExec.reply;
                    }
                    if (usingNativeTransformer)
                    {
                        std::string source = "addon:" + addonExec.meta.value("addon", addonCall.type.empty() ? std::string("auto") : addonCall.type);
                        std::lock_guard<std::mutex> lock(transformerMu_);
                        transformer_.addPreferenceFeedback(text, graphContext, reply, stripAddonToolDirective(transformerReplyRaw), 1.0f, source);
                    }
                }
                json result = graphResult;
                result["graph"] = graphResult;
                result["graphReply"] = graphResult.value("reply", "");
                result["reply"] = reply;
                result["transformerReply"] = transformerReplyRaw;
                result["graphContext"] = graphContext;
                result["gnnAlign"] = gnnHint.align;
                if (!gnnHint.annotation.empty()) result["gnnStage2"] = gnnHint.annotation;
                if (!sparkAnn.empty()) result["sparkGnnStage2"] = sparkAnn;
                if (!hint.text.empty()) {
                    result["contextHint"] = hint.text;
                    result["contextWeight"] = hint.weight;
                    if (!hint.mode.empty()) result["contextMode"] = hint.mode;
                }
                if (!imageContext.empty()) result["imageContext"] = imageContext;
                if (imageEmbCount > 0) result["imageEmbeddingCount"] = imageEmbCount;
                if (addonUsed) {
                    result["addon"] = addonExec.meta;
                    result["toolCall"] = json{{"type", addonCall.type}, {"input", addonCall.input}};
                }
                result["verify"] = verify;
                result["cached"] = cacheHit;
                result["longQuerySafeMode"] = longQuerySafeMode;
                result["gnnModuleDisabled"] = gnnModuleDisabled;
                result["contextModuleDisabled"] = contextModuleDisabled;
                result["usingOllama"] = usingOllama;
                result["transformerMode"] = backendMode;
                result["modelElapsedMs"] = (Json::Int64)modelElapsedMs;
                result["verifyElapsedMs"] = (Json::Int64)verifyElapsedMs;
                if (usingOllama)
                    result["provider"] = json{{"id", (config_.ollamaFineTuning ? "ollama+fine-tuning" : "ollama")}, {"model", config_.ollamaModel}};
                else if (usingLlamaCpp)
                    result["provider"] = json{{"id", "llamacpp"}, {"model", config_.llamaCppModel}};
                else if (usingBitNet)
                    result["provider"] = json{{"id", "bitnet"}, {"model", config_.bitnetModel}};
                else
                    result["provider"] = json{{"id", "core"}, {"model", "native-transformer"}};
                double verifyScore = verify.value("score", 0.0);
                (void)verifyScore;
                auto totalElapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - routeStarted).count();
                logRouteCompute("/api/chat", std::string("ok cached=") + (cacheHit ? "true" : "false") +
                                              " addon=" + (addonUsed ? "true" : "false") +
                                              " modelMs=" + std::to_string(modelElapsedMs) +
                                              " verifyMs=" + std::to_string(verifyElapsedMs) +
                                              " totalMs=" + std::to_string(totalElapsedMs) +
                                              " provider=" + backendMode);
                respondJson(cb, json{{"ok", true}, {"result", result}});
                onDialogCompleted(result, body);
                finalizeMetric(true);
            } catch (const std::exception &e) {
                logRouteError("/api/chat", e.what());
                respondJson(cb, json{{"ok", false}, {"error", e.what()}}, drogon::k500InternalServerError);
                finalizeMetric(false);
            } catch (...) {
                logRouteError("/api/chat", "internal error");
                respondJson(cb, json{{"ok", false}, {"error", "internal error"}}, drogon::k500InternalServerError);
                finalizeMetric(false);
            } }, {drogon::Post});

        drogon::app().registerHandler("/api/chat", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            logRouteError("/api/chat", std::string("method-not-allowed:") + req->methodString());
            respondJson(cb, json{{"ok", false}, {"error", "method not allowed"}, {"allowed", "POST"}}, drogon::k405MethodNotAllowed);
        }, {drogon::Get});

        drogon::app().registerHandler("/api/transformer/chat", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            auto routeStarted = std::chrono::steady_clock::now();
            if (config_.httpLog) std::cout << "[gateway] ENTER /api/transformer/chat" << std::endl;
            auto finalizeMetric = [&](bool ok) {
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - routeStarted).count();
                recordRouteMetric("/api/transformer/chat", elapsed, ok);
            };
            if (!authOK(req)) {
                finalizeMetric(false);
                return unauthorized(cb);
            }
            try {
                auto jsonStringOrEmpty = [](const json &obj, const char *key) -> std::string {
                    if (!obj.is_object() || !obj.contains(key) || !obj[key].is_string()) {
                        return std::string();
                    }
                    return obj[key].get<std::string>();
                };
                bool ok = true;
                json body = parseRequestBody(req, ok);
                if (!ok) {
                    finalizeMetric(false);
                    return badRequest(cb, "invalid json");
                }
                std::string text = body.value("text", body.value("message", ""));
                bool cleanedText = false;
                text = sanitizeInputText(text, cleanedText);
                if (cleanedText)
                    cleanedInputCount_.fetch_add(1, std::memory_order_relaxed);
                if (text.empty()) {
                    finalizeMetric(false);
                    return badRequest(cb, "text required");
                }
                bool simpleQuery = false;
                int tokenCount = roughTokenCount(text);
                bool longQuerySafeMode = tokenCount >= 24 || text.size() >= 360;
                if (!config_.inferenceEnabled) {
                    finalizeMetric(false);
                    return respondJson(cb, json{{"ok", false}, {"error", "inference disabled"}}, drogon::k503ServiceUnavailable);
                }
                std::string backendMode = trimCopy(config_.transformerBackend);
                std::transform(backendMode.begin(), backendMode.end(), backendMode.begin(), [](unsigned char ch)
                               { return static_cast<char>(std::tolower(ch)); });
                if (backendMode.empty())
                    backendMode = config_.usingOllama ? (config_.ollamaFineTuning ? "ollama-fine-tuning" : "ollama") : "native";
                bool usingOllama = (backendMode == "ollama" || backendMode == "ollama-fine-tuning");
                bool usingLlamaCpp = (backendMode == "llamacpp" || backendMode == "llama.cpp" || backendMode == "llama_cpp");
                bool usingBitNet = (backendMode == "bitnet");
                bool usingNativeTransformer = !(usingOllama || usingLlamaCpp || usingBitNet);
                bool gnnModuleDisabled = config_.disableGnnModule;
                bool contextModuleDisabled = config_.disableContextModule;
                bool useGraphSelector = body.value("enableGraphSelector", false);
                bool hardDisableGraphSelector = boolFrom(getEnv("AI_DISABLE_GRAPH_SELECTOR", "true"), true);
                if (hardDisableGraphSelector)
                    useGraphSelector = false;
                if (gnnModuleDisabled)
                    useGraphSelector = false;
                if (usingNativeTransformer)
                {
                    std::lock_guard<std::mutex> lock(transformerMu_);
                    auto tp = transformer_.params();
                    if (tp.value("enableDynamicReasoning", true)) {
                        int th = std::max(2, tp.value("simpleTokenThreshold", 10));
                        simpleQuery = tokenCount > 0 && tokenCount <= th;
                    }
                }
                std::string graphContext = body.value("graphContext", "");
                if (contextModuleDisabled)
                    graphContext.clear();
                auto hint = extractContextHint(body);
                if (contextModuleDisabled)
                    hint = ContextHint{};
                json graphResult;
                std::string imageContext = extractImageContext(body);
                if (contextModuleDisabled)
                    imageContext.clear();
                if (useGraphSelector && graphContext.empty() && !longQuerySafeMode) {
                    json bodyForGraph = body;
                    injectImageContext(bodyForGraph, imageContext);
                    bodyForGraph["disableAddon"] = true;
                    auto controller = pool_->getActive();
                    graphResult = controller->respond(bodyForGraph);
                }
                if (!imageContext.empty()) {
                    if (graphContext.empty()) graphContext = imageContext;
                    else graphContext = graphContext + "\n" + imageContext;
                }
                std::string graphReplyText = graphResult.is_object() ? graphResult.value("reply", "") : std::string();
                if (!graphReplyText.empty()) {
                    if (graphContext.empty()) graphContext = graphReplyText;
                    else graphContext = graphContext + "\n" + graphReplyText;
                }
                auto gnnHint = buildGnnHintSummary(graphResult, body, hint);
                hint = harmonizeHintWithGnn(hint, gnnHint.align);
                if (gnnModuleDisabled)
                    gnnHint = GnnHintSummary{};
                std::string sparkAnn;
                (void)sparkAnn;
                if (!sparkAnn.empty()) {
                    if (graphContext.empty()) graphContext = sparkAnn;
                    else graphContext = graphContext + "\n" + sparkAnn;
                }
                if (!contextModuleDisabled)
                {
                    text = applyContextHintToText(text, hint);
                    graphContext = applyContextHintToGraph(graphContext, hint);
                }
                else
                {
                    graphContext.clear();
                }
                bool cleanedGraph = false;
                graphContext = sanitizeInputText(graphContext, cleanedGraph);
                if (cleanedGraph)
                    cleanedInputCount_.fetch_add(1, std::memory_order_relaxed);
                bool addonContractEnabled = body.value("enableAddonToolContract", body.value("enableAddonTools", false));
                if (longQuerySafeMode) addonContractEnabled = false;
                if (!simpleQuery && addonContractEnabled) {
                    graphContext = attachAddonToolContract(graphContext);
                }
                int maxTokens = body.value("maxTokens", 128);
                if (longQuerySafeMode) maxTokens = std::min(maxTokens, 32);
                maxTokens = std::max(16, maxTokens);
                std::string sessionId = body.value("sessionId", "");
                std::string providerTag = std::string("provider:") + backendMode;
                std::string cacheKey = buildInferenceCacheKey(sessionId, text, providerTag + "\n" + graphContext, maxTokens);
                json tReply;
                json verify;
                bool cacheHit = false;
                int imageEmbCount = 0;
                if (db_) {
                    json cached;
                    if (db_->getInferenceCache(cacheKey, cached)) {
                        cacheHit = true;
                        tReply = cached.contains("tReply") ? cached["tReply"] : cached;
                        if (cached.contains("verify")) verify = cached["verify"];
                    }
                }
                if (cacheHit && verify.is_null()) {
                    verify = json{{"ok", true}, {"cached", true}};
                }
                if (!cacheHit) {
                    if (usingLlamaCpp || usingBitNet)
                        dispatchExternalStyleTrainStep(backendMode, body, text, gnnHint.align);
                    std::string modelGraphContext = graphContext;
                    if (usingOllama)
                    {
                        tReply = chatWithOllama(text, modelGraphContext, maxTokens);
                    }
                    else if (usingLlamaCpp)
                    {
                        tReply = chatWithLlamaCpp(text, modelGraphContext, maxTokens);
                    }
                    else if (usingBitNet)
                    {
                        tReply = chatWithBitNet(text, modelGraphContext, maxTokens);
                    }
                    else
                    {
                        std::lock_guard<std::mutex> lock(transformerMu_);
                        int dim = transformer_.params().value("dModel", 128);
                        int rounds = gnnHint.align >= 0.45 ? 2 : 1;
                        std::vector<std::vector<float>> graphEmb;
                        bool enableGraphEmbeddings = body.value("enableGraphEmbeddings", false);
                        if (enableGraphEmbeddings) {
                            graphEmb = buildGraphEmbeddings(graphResult, dim, rounds);
                        }
                        auto imageEmb = extractImageEmbeddings(body, dim);
                        if (!imageEmb.empty()) {
                            imageEmbCount = (int)imageEmb.size();
                            graphEmb.insert(graphEmb.end(), imageEmb.begin(), imageEmb.end());
                        }
                        tReply = transformer_.chat(text, modelGraphContext, graphEmb, maxTokens);
                    }
                    if (!tReply.is_object()) {
                        finalizeMetric(false);
                        return respondJson(cb, json{{"ok", false}, {"connected", false}, {"error", "disconnected"}});
                    }
                    if (longQuerySafeMode) {
                        verify = json{{"ok", false}, {"skipped", true}, {"reason", "long-query-safe-mode"}};
                    } else if (!usingNativeTransformer) {
                        verify = json{{"ok", false}, {"skipped", true}, {"reason", std::string("using-") + backendMode}};
                    } else {
                        std::lock_guard<std::mutex> lock(transformerMu_);
                        verify = transformer_.verify(text, graphContext, jsonStringOrEmpty(tReply, "reply"));
                    }
                    if (db_) {
                        db_->setInferenceCache(cacheKey, json{{"tReply", tReply}, {"verify", verify}}, 86400);
                    }
                }
                std::string transformerReplyRaw = jsonStringOrEmpty(tReply, "reply");
                std::string finalReply = stripAddonToolDirective(transformerReplyRaw);
                addon::AddonResult addonExec;
                AddonToolCall addonCall;
                bool addonUsed = false;
                if (addonContractEnabled) {
                    addonCall = parseAddonToolCall(transformerReplyRaw, text);
                    addonUsed = invokeAddonTool(addonCall, body, addonExec);
                }
                if (addonUsed) {
                    std::string followupGraphContext = buildAddonResultContext(graphContext, addonCall, addonExec);
                    if (usingLlamaCpp || usingBitNet)
                        dispatchExternalStyleTrainStep(backendMode, body, text, gnnHint.align);
                    json tReplyAfterTool;
                    if (usingOllama)
                    {
                        tReplyAfterTool = chatWithOllama(text, followupGraphContext, maxTokens);
                    }
                    else if (usingLlamaCpp)
                    {
                        tReplyAfterTool = chatWithLlamaCpp(text, followupGraphContext, maxTokens);
                    }
                    else if (usingBitNet)
                    {
                        tReplyAfterTool = chatWithBitNet(text, followupGraphContext, maxTokens);
                    }
                    else
                    {
                        std::lock_guard<std::mutex> lock(transformerMu_);
                        tReplyAfterTool = transformer_.chat(text, followupGraphContext, maxTokens);
                    }
                    std::string rewritten = stripAddonToolDirective(jsonStringOrEmpty(tReplyAfterTool, "reply"));
                    if (!rewritten.empty()) {
                        finalReply = rewritten;
                        std::string replyAfterToolRaw = jsonStringOrEmpty(tReplyAfterTool, "reply");
                        if (!replyAfterToolRaw.empty()) {
                            transformerReplyRaw = replyAfterToolRaw;
                        }
                    } else {
                        finalReply = addonExec.reply;
                    }
                    if (usingNativeTransformer)
                    {
                        std::string source = "addon:" + addonExec.meta.value("addon", addonCall.type.empty() ? std::string("auto") : addonCall.type);
                        std::lock_guard<std::mutex> lock(transformerMu_);
                        transformer_.addPreferenceFeedback(text, graphContext, finalReply, stripAddonToolDirective(transformerReplyRaw), 1.0f, source);
                    }
                }
                if (!tReply.is_object()) {
                    finalizeMetric(false);
                    return respondJson(cb, json{{"ok", false}, {"connected", false}, {"error", "disconnected"}});
                }
                if (!verify.is_object()) {
                    verify = json{{"ok", false}, {"error", "verify unavailable"}};
                }
                json result = tReply;
                result["transformerReply"] = transformerReplyRaw;
                result["reply"] = finalReply;
                result["graphContext"] = graphContext;
                result["gnnAlign"] = gnnHint.align;
                if (!gnnHint.annotation.empty()) result["gnnStage2"] = gnnHint.annotation;
                if (!sparkAnn.empty()) result["sparkGnnStage2"] = sparkAnn;
                if (!hint.text.empty()) {
                    result["contextHint"] = hint.text;
                    result["contextWeight"] = hint.weight;
                    if (!hint.mode.empty()) result["contextMode"] = hint.mode;
                }
                if (!imageContext.empty()) result["imageContext"] = imageContext;
                if (imageEmbCount > 0) result["imageEmbeddingCount"] = imageEmbCount;
                if (addonUsed) {
                    result["addon"] = addonExec.meta;
                    result["toolCall"] = json{{"type", addonCall.type}, {"input", addonCall.input}};
                }
                result["verify"] = verify;
                result["cached"] = cacheHit;
                if (!graphResult.is_null()) result["graph"] = graphResult;
                result["longQuerySafeMode"] = longQuerySafeMode;
                result["gnnModuleDisabled"] = gnnModuleDisabled;
                result["contextModuleDisabled"] = contextModuleDisabled;
                result["usingOllama"] = usingOllama;
                result["transformerMode"] = backendMode;
                if (usingOllama)
                    result["provider"] = json{{"id", "ollama"}, {"model", config_.ollamaModel}};
                else if (usingLlamaCpp)
                    result["provider"] = json{{"id", "llamacpp"}, {"model", config_.llamaCppModel}};
                else if (usingBitNet)
                    result["provider"] = json{{"id", "bitnet"}, {"model", config_.bitnetModel}};
                else
                    result["provider"] = json{{"id", "core"}, {"model", "native-transformer"}};
                double verifyScore = verify.value("score", 0.0);
                (void)verifyScore;
                logRouteCompute("/api/transformer/chat", std::string("ok cached=") + (cacheHit ? "true" : "false") + " addon=" + (addonUsed ? "true" : "false"));
                respondJson(cb, json{{"ok", true}, {"result", result}});
                finalizeMetric(true);
            } catch (const std::exception &e) {
                logRouteError("/api/transformer/chat", e.what());
                respondJson(cb, json{{"ok", false}, {"error", e.what()}}, drogon::k500InternalServerError);
                finalizeMetric(false);
            } catch (...) {
                logRouteError("/api/transformer/chat", "internal error");
                respondJson(cb, json{{"ok", false}, {"error", "internal error"}}, drogon::k500InternalServerError);
                finalizeMetric(false);
            } }, {drogon::Post});

        drogon::app().registerHandler("/api/transformer/chat", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            logRouteError("/api/transformer/chat", std::string("method-not-allowed:") + req->methodString());
            respondJson(cb, json{{"ok", false}, {"error", "method not allowed"}, {"allowed", "POST"}}, drogon::k405MethodNotAllowed);
        }, {drogon::Get});

        drogon::app().registerHandler("/api/transformer/pretrain", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            try {
                if (!authOK(req)) return unauthorized(cb);
                bool ok = true;
                json payload = parseRequestBody(req, ok);
                if (!ok) return badRequest(cb, "invalid json");
                int64_t epochs64 = 1;
                std::string parseErr;
                if (!parseBoundedInt64(payload, "epochs", 1, 1000, epochs64, parseErr))
                    return badRequest(cb, parseErr);
                int epochs = (int)epochs64;
                float lr = payload.value("lr", 0.001f);
                if (!(lr > 0.0f) || lr > 1.0f)
                    return badRequest(cb, "lr out of range");
                auto samples = parseTrainSamples(payload);
                if (samples.empty())
                    return badRequest(cb, "samples required");
                if (samples.size() > 50000)
                    return badRequest(cb, "samples too large, please batch train (<=50000)");
                uint64_t charCount = 0;
                for (const auto &s : samples)
                    charCount += (uint64_t)s.input.size() + (uint64_t)s.target.size() + (uint64_t)s.graph.size();
                uint64_t monitorId = beginTrainMonitor("/api/transformer/pretrain", json{{"epochs", epochs}, {"lr", lr}, {"samples", samples.size()}});
                touchTrainMonitor("pretrain-running", 0, 0, (uint64_t)samples.size(), charCount);
                json out;
                {
                    std::lock_guard<std::mutex> lock(transformerMu_);
                    out = transformer_.pretrain(samples, epochs, lr);
                }
                finishTrainMonitor(out.value("ok", false), out);
                if (logger_.enabled() && logger_.mode() == LoggerCXX::Mode::Release && !releaseSnapshotLogged_) {
                    logger_.log(LoggerCXX::Type::MEMORY, buildMemorySnapshotLog());
                    releaseSnapshotLogged_ = true;
                }
                respondJson(cb, json{{"ok", true}, {"result", out}, {"monitorId", monitorId}});
            } catch (const std::bad_alloc &e) {
                finishTrainMonitor(false, json::object(), std::string("bad_alloc: ") + e.what());
                logRouteError("/api/transformer/pretrain", std::string("bad_alloc: ") + e.what());
                respondJson(cb, json{{"ok", false}, {"error", "insufficient memory during pretrain"}}, drogon::k500InternalServerError);
            } catch (const std::exception &e) {
                finishTrainMonitor(false, json::object(), e.what());
                logRouteError("/api/transformer/pretrain", e.what());
                respondJson(cb, json{{"ok", false}, {"error", e.what()}}, drogon::k500InternalServerError);
            } catch (...) {
                finishTrainMonitor(false, json::object(), "internal error");
                logRouteError("/api/transformer/pretrain", "internal error");
                respondJson(cb, json{{"ok", false}, {"error", "internal error"}}, drogon::k500InternalServerError);
            } }, {drogon::Post});

        drogon::app().registerHandler("/api/external_style/train_step", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            try {
                if (!authOK(req)) return unauthorized(cb);
                bool ok = true;
                json payload = parseRequestBody(req, ok);
                if (!ok) return badRequest(cb, "invalid json");

                std::string mode = trimCopy(payload.value("provider", payload.value("backend", config_.transformerBackend)));
                std::transform(mode.begin(), mode.end(), mode.begin(), [](unsigned char c)
                               { return static_cast<char>(std::tolower(c)); });
                bool usingLlamaCpp = (mode == "llamacpp" || mode == "llama.cpp" || mode == "llama_cpp");
                bool usingBitNet = (mode == "bitnet");
                if (!usingLlamaCpp && !usingBitNet)
                    return badRequest(cb, "provider must be llamacpp or bitnet");

                std::string text = trimCopy(payload.value("text", payload.value("message", "")));
                if (text.empty())
                    return badRequest(cb, "text required");

                std::vector<std::string> keywords = parseStyleKeywords(payload, text);
                json trainPayload = buildExternalStyleTrainPayload(
                    usingBitNet,
                    payload,
                    text,
                    payload.value("observedAlign", payload.value("gnnAlign", 0.0)),
                    keywords);

                json result = usingBitNet
                                  ? postExternalStyleTrainStep("bitnet", config_.bitnetBaseUrl, config_.bitnetModel, config_.bitnetRequestTimeoutMs, trainPayload)
                                  : postExternalStyleTrainStep("llamacpp", config_.llamaCppBaseUrl, config_.llamaCppModel, config_.llamaCppRequestTimeoutMs, trainPayload);

                respondJson(cb, json{{"ok", result.value("ok", false)},
                                     {"provider", usingBitNet ? "bitnet" : "llamacpp"},
                                     {"request", trainPayload},
                                     {"result", result}});
            } catch (const std::exception &e) {
                logRouteError("/api/external_style/train_step", e.what());
                respondJson(cb, json{{"ok", false}, {"error", e.what()}}, drogon::k500InternalServerError);
            } catch (...) {
                logRouteError("/api/external_style/train_step", "internal error");
                respondJson(cb, json{{"ok", false}, {"error", "internal error"}}, drogon::k500InternalServerError);
            } }, {drogon::Post});

        drogon::app().registerHandler("/api/fine_tuning/corpus/add", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            try {
                if (!authOK(req)) return unauthorized(cb);
                bool ok = true;
                json payload = parseRequestBody(req, ok);
                if (!ok) return badRequest(cb, "invalid json");

                fs::path defaultCorpus = config_.exportDir / "fine_tuning" / "append_corpus.jsonl";
                fs::path corpusPath = defaultCorpus;
                if (payload.contains("appendCorpusPath") && payload["appendCorpusPath"].is_string()) {
                    auto raw = trimCopy(payload["appendCorpusPath"].get<std::string>());
                    if (!raw.empty()) {
                        fs::path p(raw);
                        corpusPath = p.is_relative() ? fs::absolute(config_.baseDir / p) : fs::absolute(p);
                    }
                }
                std::error_code ec;
                fs::create_directories(corpusPath.parent_path(), ec);

                json items = json::array();
                if (payload.contains("items") && payload["items"].is_array()) {
                    items = payload["items"];
                } else {
                    std::string q = payload.value("question", "");
                    std::string a = payload.value("answer", "");
                    std::string s = payload.value("style", "default");
                    if (!trimCopy(q).empty() || !trimCopy(a).empty()) {
                        items.push_back(json{{"question", q}, {"answer", a}, {"style", s}});
                    }
                }

                if (!items.is_array() || items.empty()) {
                    return badRequest(cb, "items required");
                }

                uint64_t added = 0;
                uint64_t charCount = 0;
                std::ofstream out(corpusPath, std::ios::out | std::ios::app | std::ios::binary);
                if (!out.is_open()) {
                    return respondJson(cb, json{{"ok", false}, {"error", "cannot open corpus file"}}, drogon::k500InternalServerError);
                }
                for (const auto &it : items) {
                    if (!it.is_object())
                        continue;
                    std::string q = trimCopy(it.value("question", ""));
                    std::string a = trimCopy(it.value("answer", ""));
                    std::string s = trimCopy(it.value("style", "default"));
                    if (q.empty() || a.empty())
                        continue;
                    auto gnnKeywords = deriveGnnStyleKeywords(q + " " + a, 8);
                    std::string questionAug = q;
                    if (!gnnKeywords.empty())
                    {
                        std::ostringstream kw;
                        kw << "[GNN关键词:";
                        for (size_t i = 0; i < gnnKeywords.size(); i++)
                        {
                            if (i)
                                kw << ",";
                            kw << gnnKeywords[i];
                        }
                        kw << "] ";
                        questionAug = kw.str() + q;
                    }
                    json row{{"question", questionAug}, {"questionRaw", q}, {"answer", a}, {"style", s.empty() ? "default" : s}, {"gnnKeywords", gnnKeywords}};
                    std::string line = row.dump();
                    out << line << "\n";
                    added += 1;
                    charCount += (uint64_t)q.size() + (uint64_t)a.size();
                }
                out.flush();

                if (added == 0) {
                    return badRequest(cb, "no valid qa pairs");
                }

                uint64_t totalLines = 0;
                {
                    std::ifstream in(corpusPath, std::ios::in | std::ios::binary);
                    std::string line;
                    while (std::getline(in, line)) {
                        if (!trimCopy(line).empty()) totalLines += 1;
                    }
                }

                respondJson(cb, json{{"ok", true}, {"result", json{{"path", corpusPath.string()}, {"added", added}, {"total", totalLines}, {"chars", charCount}}}});
            } catch (const std::exception &e) {
                logRouteError("/api/fine_tuning/corpus/add", e.what());
                respondJson(cb, json{{"ok", false}, {"error", e.what()}}, drogon::k500InternalServerError);
            } catch (...) {
                logRouteError("/api/fine_tuning/corpus/add", "internal error");
                respondJson(cb, json{{"ok", false}, {"error", "internal error"}}, drogon::k500InternalServerError);
            } }, {drogon::Post});

        drogon::app().registerHandler("/api/fine_tuning/run", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            try {
                if (!authOK(req)) return unauthorized(cb);
                bool ok = true;
                json payload = parseRequestBody(req, ok);
                if (!ok) return badRequest(cb, "invalid json");

                {
                    std::lock_guard<std::mutex> lock(fineTuningMu_);
                    if (fineTuningRunning_) {
                        return respondJson(cb, json{{"ok", false}, {"error", "fine-tuning already running"}}, drogon::k409Conflict);
                    }
                    fineTuningRunning_ = true;
                }

                auto guardRunning = std::shared_ptr<void>(nullptr, [this](void *) {
                    std::lock_guard<std::mutex> lock(this->fineTuningMu_);
                    this->fineTuningRunning_ = false;
                });

                fs::path defaultCorpus = config_.exportDir / "fine_tuning" / "append_corpus.jsonl";
                fs::path defaultOutDir = config_.exportDir / "fine_tuning" / "checkpoints";

                fs::path corpusPath = defaultCorpus;
                fs::path outputDir = defaultOutDir;
                std::string ollamaModel = payload.value("ollamaModel", config_.ollamaModel.empty() ? std::string("gpt-oss:20b") : config_.ollamaModel);
                std::string hfModel = payload.value("hfModel", std::string("gpt-oss:20b"));

                if (payload.contains("appendCorpusPath") && payload["appendCorpusPath"].is_string()) {
                    auto raw = trimCopy(payload["appendCorpusPath"].get<std::string>());
                    if (!raw.empty()) {
                        fs::path p(raw);
                        corpusPath = p.is_relative() ? fs::absolute(config_.baseDir / p) : fs::absolute(p);
                    }
                }
                if (payload.contains("outputDir") && payload["outputDir"].is_string()) {
                    auto raw = trimCopy(payload["outputDir"].get<std::string>());
                    if (!raw.empty()) {
                        fs::path p(raw);
                        outputDir = p.is_relative() ? fs::absolute(config_.baseDir / p) : fs::absolute(p);
                    }
                }

                std::error_code ec;
                fs::create_directories(corpusPath.parent_path(), ec);
                fs::create_directories(outputDir, ec);

                uint64_t appendedNow = 0;
                {
                    json incomingItems = json::array();
                    if (payload.contains("items") && payload["items"].is_array()) {
                        incomingItems = payload["items"];
                    }
                    if (payload.contains("question") || payload.contains("answer")) {
                        incomingItems.push_back(json{{"question", payload.value("question", "")},
                                                     {"answer", payload.value("answer", "")},
                                                     {"style", payload.value("style", "default")}});
                    }

                    if (incomingItems.is_array() && !incomingItems.empty()) {
                        std::ofstream out(corpusPath, std::ios::out | std::ios::app | std::ios::binary);
                        if (!out.is_open()) {
                            return respondJson(cb, json{{"ok", false}, {"error", "cannot open corpus file"}}, drogon::k500InternalServerError);
                        }
                        for (const auto &it : incomingItems) {
                            if (!it.is_object())
                                continue;
                            std::string q = trimCopy(it.value("question", ""));
                            std::string a = trimCopy(it.value("answer", ""));
                            std::string s = trimCopy(it.value("style", "default"));
                            if (q.empty() || a.empty())
                                continue;
                            auto gnnKeywords = deriveGnnStyleKeywords(q + " " + a, 8);
                            std::string questionAug = q;
                            if (!gnnKeywords.empty())
                            {
                                std::ostringstream kw;
                                kw << "[GNN关键词:";
                                for (size_t i = 0; i < gnnKeywords.size(); i++)
                                {
                                    if (i)
                                        kw << ",";
                                    kw << gnnKeywords[i];
                                }
                                kw << "] ";
                                questionAug = kw.str() + q;
                            }
                            out << json{{"question", questionAug}, {"questionRaw", q}, {"answer", a}, {"style", s.empty() ? "default" : s}, {"gnnKeywords", gnnKeywords}}.dump() << "\n";
                            appendedNow += 1;
                        }
                        out.flush();
                    }
                }

                if (!fs::exists(corpusPath)) {
                    return badRequest(cb, std::string("append corpus not found: ") + corpusPath.string());
                }

                int64_t selfPlayPairs64 = 0;
                int64_t epochs64 = 1;
                std::string parseErr;
                if (!parseBoundedInt64(payload, "selfPlayPairs", 0, 100000, selfPlayPairs64, parseErr)) {
                    return badRequest(cb, parseErr);
                }
                if (!parseBoundedInt64(payload, "epochs", 1, 1000, epochs64, parseErr)) {
                    return badRequest(cb, parseErr);
                }
                double lr = payload.value("lr", 2e-5);
                if (!(lr > 0.0) || lr > 1.0) {
                    return badRequest(cb, "lr out of range");
                }

                uint64_t monitorId = beginTrainMonitor(
                    "/api/fine_tuning/run",
                    json{{"appendCorpusPath", corpusPath.string()},
                         {"outputDir", outputDir.string()},
                        {"appendedNow", appendedNow},
                         {"ollamaModel", ollamaModel},
                         {"hfModel", hfModel},
                         {"selfPlayPairs", selfPlayPairs64},
                         {"epochs", epochs64},
                         {"lr", lr}});
                touchTrainMonitor("fine-tuning-init", 0, 0, 0, 0);

                char errBuf[4096] = {0};
                if (!transformer_ollama_ft_bridge_set_module("transformer_ollama_fine_tuning", errBuf, sizeof(errBuf))) {
                    std::string err = errBuf[0] ? std::string(errBuf) : std::string("bridge_set_module failed");
                    finishTrainMonitor(false, json::object(), err);
                    return respondJson(cb, json{{"ok", false}, {"error", err}}, drogon::k500InternalServerError);
                }
                if (!transformer_ollama_ft_bridge_init(errBuf, sizeof(errBuf))) {
                    std::string err = errBuf[0] ? std::string(errBuf) : std::string("bridge_init failed");
                    finishTrainMonitor(false, json::object(), err);
                    return respondJson(cb, json{{"ok", false}, {"error", err}}, drogon::k500InternalServerError);
                }

                touchTrainMonitor("fine-tuning-running", 0, 0, (uint64_t)selfPlayPairs64, 0);
                int rc = transformer_ollama_ft_bridge_train(
                    corpusPath.string().c_str(),
                    outputDir.string().c_str(),
                    ollamaModel.c_str(),
                    hfModel.c_str(),
                    (int)selfPlayPairs64,
                    (int)epochs64,
                    lr,
                    errBuf,
                    sizeof(errBuf));

                if (!rc) {
                    std::string err = errBuf[0] ? std::string(errBuf) : std::string("fine-tuning failed");
                    finishTrainMonitor(false, json::object(), err);
                    return respondJson(cb, json{{"ok", false}, {"error", err}, {"monitorId", monitorId}}, drogon::k500InternalServerError);
                }

                json result{{"ok", true},
                            {"appendCorpusPath", corpusPath.string()},
                            {"outputDir", outputDir.string()},
                            {"appendedNow", appendedNow},
                            {"ollamaModel", ollamaModel},
                            {"hfModel", hfModel},
                            {"epochs", epochs64},
                            {"selfPlayPairs", selfPlayPairs64},
                            {"lr", lr}};
                finishTrainMonitor(true, result);
                respondJson(cb, json{{"ok", true}, {"result", result}, {"monitorId", monitorId}});
            } catch (const std::exception &e) {
                finishTrainMonitor(false, json::object(), e.what());
                logRouteError("/api/fine_tuning/run", e.what());
                respondJson(cb, json{{"ok", false}, {"error", e.what()}}, drogon::k500InternalServerError);
            } catch (...) {
                finishTrainMonitor(false, json::object(), "internal error");
                logRouteError("/api/fine_tuning/run", "internal error");
                respondJson(cb, json{{"ok", false}, {"error", "internal error"}}, drogon::k500InternalServerError);
            } }, {drogon::Post});

        drogon::app().registerHandler("/api/transformer/joint_train", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            try {
                if (!authOK(req)) return unauthorized(cb);
                bool ok = true;
                json payload = parseRequestBody(req, ok);
                if (!ok) return badRequest(cb, "invalid json");
                int64_t epochs64 = 1;
                std::string parseErr;
                if (!parseBoundedInt64(payload, "epochs", 1, 1000, epochs64, parseErr))
                    return badRequest(cb, parseErr);
                int epochs = (int)epochs64;
                float lr = payload.value("lr", 0.001f);
                float gw = payload.value("graphWeight", 0.35f);
                if (!(lr > 0.0f) || lr > 1.0f)
                    return badRequest(cb, "lr out of range");
                if (gw < 0.0f || gw > 1.0f)
                    return badRequest(cb, "graphWeight out of range");
                auto samples = parseTrainSamples(payload);
                if (samples.empty())
                    return badRequest(cb, "samples required");
                if (samples.size() > 50000)
                    return badRequest(cb, "samples too large, please batch train (<=50000)");
                uint64_t charCount = 0;
                for (const auto &s : samples)
                    charCount += (uint64_t)s.input.size() + (uint64_t)s.target.size() + (uint64_t)s.graph.size();
                uint64_t monitorId = beginTrainMonitor("/api/transformer/joint_train", json{{"epochs", epochs}, {"lr", lr}, {"graphWeight", gw}, {"samples", samples.size()}});
                touchTrainMonitor("joint-train-running", 0, 0, (uint64_t)samples.size(), charCount);
                json out;
                {
                    std::lock_guard<std::mutex> lock(transformerMu_);
                    out = transformer_.jointTrain(samples, epochs, lr, gw);
                }
                finishTrainMonitor(out.value("ok", false), out);
                if (logger_.enabled() && logger_.mode() == LoggerCXX::Mode::Release && !releaseSnapshotLogged_) {
                    logger_.log(LoggerCXX::Type::MEMORY, buildMemorySnapshotLog());
                    releaseSnapshotLogged_ = true;
                }
                respondJson(cb, json{{"ok", true}, {"result", out}, {"monitorId", monitorId}});
            } catch (const std::bad_alloc &e) {
                finishTrainMonitor(false, json::object(), std::string("bad_alloc: ") + e.what());
                logRouteError("/api/transformer/joint_train", std::string("bad_alloc: ") + e.what());
                respondJson(cb, json{{"ok", false}, {"error", "insufficient memory during joint_train"}}, drogon::k500InternalServerError);
            } catch (const std::exception &e) {
                finishTrainMonitor(false, json::object(), e.what());
                logRouteError("/api/transformer/joint_train", e.what());
                respondJson(cb, json{{"ok", false}, {"error", e.what()}}, drogon::k500InternalServerError);
            } catch (...) {
                finishTrainMonitor(false, json::object(), "internal error");
                logRouteError("/api/transformer/joint_train", "internal error");
                respondJson(cb, json{{"ok", false}, {"error", "internal error"}}, drogon::k500InternalServerError);
            } }, {drogon::Post});

        drogon::app().registerHandler("/api/transformer/ga_optimize", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            try {
                if (!authOK(req)) return unauthorized(cb);
                bool ok = true;
                json payload = parseRequestBody(req, ok);
                if (!ok) return badRequest(cb, "invalid json");
                int64_t generations64 = 3;
                int64_t population64 = 6;
                std::string parseErr;
                if (!parseBoundedInt64(payload, "generations", 1, 1000, generations64, parseErr))
                    return badRequest(cb, parseErr);
                if (!parseBoundedInt64(payload, "population", 2, 2048, population64, parseErr))
                    return badRequest(cb, parseErr);
                int generations = (int)generations64;
                int population = (int)population64;
                auto samples = parseTrainSamples(payload);
                if (samples.empty())
                    return badRequest(cb, "samples required");
                uint64_t monitorId = beginTrainMonitor("/api/transformer/ga_optimize", json{{"generations", generations}, {"population", population}, {"samples", samples.size()}});
                touchTrainMonitor("ga-optimize-running", 0, 0, (uint64_t)samples.size(), 0);
                json out;
                {
                    std::lock_guard<std::mutex> lock(transformerMu_);
                    out = transformer_.optimizeGA(samples, generations, population);
                }
                finishTrainMonitor(out.value("ok", false), out);
                if (logger_.enabled() && logger_.mode() == LoggerCXX::Mode::Release && !releaseSnapshotLogged_) {
                    logger_.log(LoggerCXX::Type::MEMORY, buildMemorySnapshotLog());
                    releaseSnapshotLogged_ = true;
                }
                respondJson(cb, json{{"ok", true}, {"result", out}, {"monitorId", monitorId}});
            } catch (const std::bad_alloc &e) {
                finishTrainMonitor(false, json::object(), std::string("bad_alloc: ") + e.what());
                logRouteError("/api/transformer/ga_optimize", std::string("bad_alloc: ") + e.what());
                respondJson(cb, json{{"ok", false}, {"error", "insufficient memory during ga_optimize"}}, drogon::k500InternalServerError);
            } catch (const std::exception &e) {
                finishTrainMonitor(false, json::object(), e.what());
                logRouteError("/api/transformer/ga_optimize", e.what());
                respondJson(cb, json{{"ok", false}, {"error", e.what()}}, drogon::k500InternalServerError);
            } catch (...) {
                finishTrainMonitor(false, json::object(), "internal error");
                logRouteError("/api/transformer/ga_optimize", "internal error");
                respondJson(cb, json{{"ok", false}, {"error", "internal error"}}, drogon::k500InternalServerError);
            } }, {drogon::Post});

        drogon::app().registerHandler("/api/transformer/verify", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            bool ok = true;
            json payload = parseRequestBody(req, ok);
            if (!ok) return badRequest(cb, "invalid json");
            std::string text = payload.value("text", "");
            std::string graphContext = payload.value("graphContext", "");
            std::string reply = payload.value("reply", "");
            json out;
            {
                std::lock_guard<std::mutex> lock(transformerMu_);
                out = transformer_.verify(text, graphContext, reply);
            }
            respondJson(cb, json{{"ok", true}, {"result", out}}); }, {drogon::Post});

        drogon::app().registerHandler("/api/transformer/feedback", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            bool ok = true;
            json payload = parseRequestBody(req, ok);
            if (!ok) return badRequest(cb, "invalid json");
            std::string text = payload.value("text", payload.value("prompt", ""));
            std::string graphContext = payload.value("graphContext", "");
            std::string reply = payload.value("reply", payload.value("output", ""));
            std::string chosen = payload.value("chosen", reply);
            std::string rejected = payload.value("rejected", "");
            std::string source = payload.value("source", "api");
            float score = payload.value("score", 0.0f);
            if (payload.contains("accepted")) {
                score = payload.value("accepted", false) ? 1.0f : -1.0f;
            }
            if (payload.contains("addon") && payload["addon"].is_object()) {
                float addonScale = 0.35f;
                {
                    std::lock_guard<std::mutex> lock(transformerMu_);
                    addonScale = transformer_.params().value("addonRewardScale", 0.35f);
                }
                bool addonOk = true;
                const auto &a = payload["addon"];
                if (a.contains("ok") && a["ok"].is_boolean()) addonOk = a["ok"].get<bool>();
                if (source == "api") {
                    std::string n = a.value("name", a.value("addon", ""));
                    std::string t = a.value("type", "");
                    if (!n.empty()) source = "addon:" + n;
                    else if (!t.empty()) source = "addonType:" + t;
                }
                score += addonOk ? addonScale : -addonScale;
            }
            bool trainNow = payload.value("trainNow", false);
            int steps = payload.value("steps", 1);
            if (chosen.empty()) return badRequest(cb, "chosen/reply required");
            json stats = json::object();
            json train = json::object();
            {
                std::lock_guard<std::mutex> lock(transformerMu_);
                transformer_.addFeedback(chosen, score);
                transformer_.addPreferenceFeedback(text, graphContext, chosen, rejected, score, source);
                if (trainNow) {
                    train = transformer_.trainFromFeedback(std::max(1, steps));
                }
                stats = transformer_.rlhfStats();
            }
            json out{{"ok", true}, {"result", stats}};
            if (trainNow) out["train"] = train;
            respondJson(cb, out); }, {drogon::Post});

        drogon::app().registerHandler("/api/transformer/feedback/train", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            bool ok = true;
            json payload = parseRequestBody(req, ok);
            if (!ok) return badRequest(cb, "invalid json");
            int64_t steps64 = 1;
            std::string parseErr;
            if (!parseBoundedInt64(payload, "steps", 1, 1000000, steps64, parseErr))
                return badRequest(cb, parseErr);
            int steps = (int)steps64;
            uint64_t monitorId = beginTrainMonitor("/api/transformer/feedback/train", json{{"steps", steps}});
            touchTrainMonitor("feedback-train-running", 0, 0, 0, 0);
            json out;
            {
                std::lock_guard<std::mutex> lock(transformerMu_);
                out = transformer_.trainFromFeedback(steps);
            }
            finishTrainMonitor(out.value("ok", false), out);
            respondJson(cb, json{{"ok", true}, {"result", out}, {"monitorId", monitorId}}); }, {drogon::Post});

        drogon::app().registerHandler("/api/transformer/feedback/status", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            json out;
            {
                std::lock_guard<std::mutex> lock(transformerMu_);
                out = transformer_.rlhfStats();
            }
            respondJson(cb, json{{"ok", true}, {"result", out}}); }, {drogon::Get});

        drogon::app().registerHandler("/api/transformer/params", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            if (req->method() == drogon::Get) {
                json out;
                {
                    std::lock_guard<std::mutex> lock(transformerMu_);
                    out = transformer_.params();
                }
                respondJson(cb, json{{"ok", true}, {"params", out}});
                return;
            }
            bool ok = true;
            json payload = parseRequestBody(req, ok);
            if (!ok) return badRequest(cb, "invalid json");
            {
                std::lock_guard<std::mutex> lock(transformerMu_);
                transformer_.applyParams(payload);
            }
            json out;
            {
                std::lock_guard<std::mutex> lock(transformerMu_);
                out = transformer_.params();
            }
            respondJson(cb, json{{"ok", true}, {"params", out}}); }, {drogon::Get, drogon::Post});

        drogon::app().registerHandler("/api/transformer/checkpoint/save", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            try {
                if (!authOK(req)) return unauthorized(cb);
                bool ok = true;
                json payload = parseRequestBody(req, ok);
                if (!ok) payload = json::object();
                std::string path = payload.value("path", std::string("runtime_store/transformer_checkpoint.json"));
                json out;
                {
                    std::lock_guard<std::mutex> lock(transformerMu_);
                    out = transformer_.saveCheckpoint(path);
                }
                auto status = out.value("ok", false) ? drogon::k200OK : drogon::k500InternalServerError;
                respondJson(cb, out, status);
            } catch (const std::exception &e) {
                logRouteError("/api/transformer/checkpoint/save", e.what());
                respondJson(cb, json{{"ok", false}, {"error", e.what()}}, drogon::k500InternalServerError);
            } catch (...) {
                logRouteError("/api/transformer/checkpoint/save", "internal error");
                respondJson(cb, json{{"ok", false}, {"error", "internal error"}}, drogon::k500InternalServerError);
            } }, {drogon::Post});

        drogon::app().registerHandler("/api/transformer/checkpoint/load", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            try {
                if (!authOK(req)) return unauthorized(cb);
                bool ok = true;
                json payload = parseRequestBody(req, ok);
                if (!ok) payload = json::object();
                std::string path = payload.value("path", std::string("runtime_store/transformer_checkpoint.json"));
                json out;
                {
                    std::lock_guard<std::mutex> lock(transformerMu_);
                    out = transformer_.loadCheckpoint(path);
                }
                auto status = out.value("ok", false) ? drogon::k200OK : drogon::k500InternalServerError;
                respondJson(cb, out, status);
            } catch (const std::exception &e) {
                logRouteError("/api/transformer/checkpoint/load", e.what());
                respondJson(cb, json{{"ok", false}, {"error", e.what()}}, drogon::k500InternalServerError);
            } catch (...) {
                logRouteError("/api/transformer/checkpoint/load", "internal error");
                respondJson(cb, json{{"ok", false}, {"error", "internal error"}}, drogon::k500InternalServerError);
            } }, {drogon::Post});

        drogon::app().registerHandler("/api/model/lifecycle", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            respondJson(cb, modelLifecycle_.status()); }, {drogon::Get});

        drogon::app().registerHandler("/api/model/compress", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            bool ok = true;
            json payload = parseRequestBody(req, ok);
            if (!ok) return badRequest(cb, "invalid json");
            json out = modelLifecycle_.compressPlan(payload);
            if (payload.value("applyTransformerParams", false)) {
                json patch = json::object();
                std::string quant = payload.value("quant", std::string(""));
                double pruneRatio = payload.value("pruneRatio", 0.0);
                if (quant == "int8") patch["dModel"] = 96;
                if (quant == "int4") patch["dModel"] = 64;
                if (pruneRatio >= 0.2) patch["nLayers"] = 2;
                if (!patch.empty()) {
                    std::lock_guard<std::mutex> lock(transformerMu_);
                    transformer_.applyParams(patch);
                    out["transformerPatch"] = patch;
                }
            }
            respondJson(cb, out); }, {drogon::Post});

        drogon::app().registerHandler("/api/model/explain", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            bool ok = true;
            json payload = parseRequestBody(req, ok);
            if (!ok) return badRequest(cb, "invalid json");
            std::string text = payload.value("text", "");
            std::string graphContext = payload.value("graphContext", "");
            std::string reply = payload.value("reply", "");
            if (reply.empty() && !text.empty()) {
                std::lock_guard<std::mutex> lock(transformerMu_);
                auto chatResult = transformer_.chat(text, graphContext, payload.value("maxTokens", 120));
                reply = chatResult.value("reply", "");
                payload["reply"] = reply;
                payload["generatedByTransformer"] = true;
            }
            json out = modelLifecycle_.explainOutput(payload);
            if (!reply.empty()) {
                json verify;
                {
                    std::lock_guard<std::mutex> lock(transformerMu_);
                    verify = transformer_.verify(text, graphContext, reply);
                }
                out["result"]["verify"] = verify;
            }
            respondJson(cb, out); }, {drogon::Post});

        drogon::app().registerHandler("/api/model/deploy", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            bool ok = true;
            json payload = parseRequestBody(req, ok);
            if (!ok) return badRequest(cb, "invalid json");
            respondJson(cb, modelLifecycle_.deployTarget(payload)); }, {drogon::Post});

        drogon::app().registerHandler("/api/model/update", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            bool ok = true;
            json payload = parseRequestBody(req, ok);
            if (!ok) return badRequest(cb, "invalid json");
            respondJson(cb, modelLifecycle_.applyOnlineUpdate(payload)); }, {drogon::Post});

        drogon::app().registerHandler("/api/cluster/status", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            respondJson(cb, clusterManager_.status()); }, {drogon::Get});

        drogon::app().registerHandler("/api/cluster/nodes", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            bool ok = true;
            json payload = parseRequestBody(req, ok);
            if (!ok) return badRequest(cb, "invalid json");
            respondJson(cb, clusterManager_.updateNodes(payload)); }, {drogon::Post});

        drogon::app().registerHandler("/api/cluster/route", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            bool ok = true;
            json payload = parseRequestBody(req, ok);
            if (!ok) return badRequest(cb, "invalid json");
            respondJson(cb, clusterManager_.pickNode(payload)); }, {drogon::Post});

        drogon::app().registerHandler("/api/cluster/feedback", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            bool ok = true;
            json payload = parseRequestBody(req, ok);
            if (!ok) return badRequest(cb, "invalid json");
            respondJson(cb, clusterManager_.feedback(payload)); }, {drogon::Post});

        drogon::app().registerHandler("/api/dataset/catalog", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            respondJson(cb, datasetCatalog_.list()); }, {drogon::Get});

        drogon::app().registerHandler("/api/data/governance", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            respondJson(cb, datasetCatalog_.governance()); }, {drogon::Get});

        drogon::app().registerHandler("/api/dataset/register", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            bool ok = true;
            json payload = parseRequestBody(req, ok);
            if (!ok) return badRequest(cb, "invalid json");
            respondJson(cb, datasetCatalog_.registerDataset(payload)); }, {drogon::Post});

        drogon::app().registerHandler("/api/dataset/activate", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            bool ok = true;
            json payload = parseRequestBody(req, ok);
            if (!ok) return badRequest(cb, "invalid json");
            respondJson(cb, datasetCatalog_.activate(payload)); }, {drogon::Post});

        drogon::app().registerHandler("/api/data/collect", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            auto routeStarted = std::chrono::steady_clock::now();
            auto finalizeMetric = [&](bool okFlag) {
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - routeStarted).count();
                recordRouteMetric("/api/data/collect", elapsed, okFlag);
            };
            if (!authOK(req)) {
                finalizeMetric(false);
                return unauthorized(cb);
            }
            try {
                bool ok = true;
                json payload = parseRequestBody(req, ok);
                if (!ok) {
                    finalizeMetric(false);
                    return badRequest(cb, "invalid json");
                }
                auto out = datasetCatalog_.collectData(payload);
                finalizeMetric(out.value("ok", false));
                respondJson(cb, out);
            } catch (const std::exception &e) {
                logRouteError("/api/data/collect", e.what());
                finalizeMetric(false);
                respondJson(cb, json{{"ok", false}, {"error", e.what()}}, drogon::k500InternalServerError);
            } catch (...) {
                logRouteError("/api/data/collect", "internal error");
                finalizeMetric(false);
                respondJson(cb, json{{"ok", false}, {"error", "internal error"}}, drogon::k500InternalServerError);
            } }, {drogon::Post});

        drogon::app().registerHandler("/api/data/cleaning/profile", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            auto routeStarted = std::chrono::steady_clock::now();
            auto finalizeMetric = [&](bool okFlag) {
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - routeStarted).count();
                recordRouteMetric("/api/data/cleaning/profile", elapsed, okFlag);
            };
            if (!authOK(req)) {
                finalizeMetric(false);
                return unauthorized(cb);
            }
            try {
                if (req->method() == drogon::Get) {
                    auto out = datasetCatalog_.list();
                    finalizeMetric(out.value("ok", false));
                    respondJson(cb, out);
                    return;
                }
                bool ok = true;
                json payload = parseRequestBody(req, ok);
                if (!ok) {
                    finalizeMetric(false);
                    return badRequest(cb, "invalid json");
                }
                auto out = datasetCatalog_.updateCleaningProfile(payload);
                finalizeMetric(out.value("ok", false));
                respondJson(cb, out);
            } catch (const std::exception &e) {
                logRouteError("/api/data/cleaning/profile", e.what());
                finalizeMetric(false);
                respondJson(cb, json{{"ok", false}, {"error", e.what()}}, drogon::k500InternalServerError);
            } catch (...) {
                logRouteError("/api/data/cleaning/profile", "internal error");
                finalizeMetric(false);
                respondJson(cb, json{{"ok", false}, {"error", "internal error"}}, drogon::k500InternalServerError);
            } }, {drogon::Get, drogon::Post});

        drogon::app().registerHandler("/api/spider/autonomy/status", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            respondJson(cb, spiderAutonomy_.status()); }, {drogon::Get});

        drogon::app().registerHandler("/api/spider/autonomy/adapt", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            bool ok = true;
            json payload = parseRequestBody(req, ok);
            if (!ok) return badRequest(cb, "invalid json");
            respondJson(cb, spiderAutonomy_.adapt(payload, getMonitoringStats())); }, {drogon::Post});

        drogon::app().registerHandler("/api/optimizer/autonomy/status", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            respondJson(cb, optimizerAutonomy_.status()); }, {drogon::Get});

        drogon::app().registerHandler("/api/optimizer/autonomy/iterate", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            bool ok = true;
            json payload = parseRequestBody(req, ok);
            if (!ok) return badRequest(cb, "invalid json");
            json currentParams;
            {
                std::lock_guard<std::mutex> lock(transformerMu_);
                currentParams = transformer_.params();
            }
            json out = optimizerAutonomy_.iterate(payload, getMonitoringStats(), currentParams);
            if (payload.value("apply", true) && out.value("ok", false) && out.contains("result") && out["result"].contains("transformerPatch")) {
                json patch = out["result"]["transformerPatch"];
                if (patch.is_object() && !patch.empty()) {
                    std::lock_guard<std::mutex> lock(transformerMu_);
                    transformer_.applyParams(patch);
                    out["result"]["applied"] = patch;
                }
            }
            respondJson(cb, out); }, {drogon::Post});

        drogon::app().registerHandler("/api/perf/profile", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            bool ok = true;
            json payload = parseRequestBody(req, ok);
            if (!ok) return badRequest(cb, "invalid json");
            json currentParams;
            {
                std::lock_guard<std::mutex> lock(transformerMu_);
                currentParams = transformer_.params();
            }
            json out = optimizerAutonomy_.applyPerfProfile(payload, currentParams);
            if (out.value("ok", false) && out.contains("result") && out["result"].contains("transformerPatch")) {
                json patch = out["result"]["transformerPatch"];
                if (patch.is_object() && !patch.empty()) {
                    std::lock_guard<std::mutex> lock(transformerMu_);
                    transformer_.applyParams(patch);
                    out["result"]["applied"] = patch;
                }
            }
            respondJson(cb, out); }, {drogon::Post});

        drogon::app().registerHandler("/api/gnn/upgrade", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            bool ok = true;
            json payload = parseRequestBody(req, ok);
            if (!ok) return badRequest(cb, "invalid json");
            json currentParams;
            {
                std::lock_guard<std::mutex> lock(transformerMu_);
                currentParams = transformer_.params();
            }
            json out = optimizerAutonomy_.proposeGnnUpgrade(payload, currentParams);
            if (out.value("ok", false) && out.contains("result") && out["result"].contains("transformerPatch")) {
                json patch = out["result"]["transformerPatch"];
                if (patch.is_object() && !patch.empty()) {
                    std::lock_guard<std::mutex> lock(transformerMu_);
                    transformer_.applyParams(patch);
                    out["result"]["applied"] = patch;
                }
            }
            respondJson(cb, out); }, {drogon::Post});

        drogon::app().registerHandler("/api/transformer/upgrade", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            bool ok = true;
            json payload = parseRequestBody(req, ok);
            if (!ok) return badRequest(cb, "invalid json");
            json currentParams;
            {
                std::lock_guard<std::mutex> lock(transformerMu_);
                currentParams = transformer_.params();
            }
            json out = optimizerAutonomy_.proposeTransformerUpgrade(payload, currentParams);
            if (out.value("ok", false) && out.contains("result") && out["result"].contains("transformerPatch")) {
                json patch = out["result"]["transformerPatch"];
                if (patch.is_object() && !patch.empty()) {
                    std::lock_guard<std::mutex> lock(transformerMu_);
                    transformer_.applyParams(patch);
                    out["result"]["applied"] = patch;
                }
            }
            respondJson(cb, out); }, {drogon::Post});

        drogon::app().registerHandler("/api/transformer/modernize", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            bool ok = true;
            json payload = parseRequestBody(req, ok);
            if (!ok) return badRequest(cb, "invalid json");
            json currentParams;
            {
                std::lock_guard<std::mutex> lock(transformerMu_);
                currentParams = transformer_.params();
            }
            json out = optimizerAutonomy_.modernizeTransformer(payload, currentParams);
            if (out.value("ok", false) && out.contains("result") && out["result"].contains("transformerPatch")) {
                json patch = out["result"]["transformerPatch"];
                if (patch.is_object() && !patch.empty()) {
                    std::lock_guard<std::mutex> lock(transformerMu_);
                    transformer_.applyParams(patch);
                    out["result"]["applied"] = patch;
                }
            }
            respondJson(cb, out); }, {drogon::Post});

        drogon::app().registerHandler("/api/runtime/features", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            respondJson(cb, json{{"ok", true}, {"features", getRuntimeFeatureState()}}); }, {drogon::Get});

        drogon::app().registerHandler("/api/provider/capabilities", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            std::string backendMode = trimCopy(config_.transformerBackend);
            std::transform(backendMode.begin(), backendMode.end(), backendMode.begin(), [](unsigned char ch)
                           { return static_cast<char>(std::tolower(ch)); });
            if (backendMode.empty())
                backendMode = config_.usingOllama ? (config_.ollamaFineTuning ? "ollama-fine-tuning" : "ollama") : "native";
            std::string defaultProvider = "off";
            if (config_.inferenceEnabled)
            {
                if (backendMode == "ollama" || backendMode == "ollama-fine-tuning")
                    defaultProvider = "ollama";
                else if (backendMode == "llamacpp" || backendMode == "llama.cpp" || backendMode == "llama_cpp")
                    defaultProvider = "llamacpp";
                else if (backendMode == "bitnet")
                    defaultProvider = "bitnet";
                else
                    defaultProvider = "core";
            }
            json out = {
                {"ok", true},
                {"result", {
                    {"providers", json::array({
                        json{{"id", "core"}, {"mode", "backend-mainchain"}, {"supports", json::array({"chat", "transformer-chat", "verify"})}},
                        json{{"id", "ollama"}, {"mode", "backend-adapter"}, {"supports", json::array({"chat", "transformer-chat"})}, {"model", config_.ollamaModel}},
                        json{{"id", "llamacpp"}, {"mode", "backend-adapter"}, {"supports", json::array({"chat", "transformer-chat"})}, {"model", config_.llamaCppModel}},
                        json{{"id", "bitnet"}, {"mode", "backend-adapter"}, {"supports", json::array({"chat", "transformer-chat"})}, {"model", config_.bitnetModel}},
                        json{{"id", "openclaw"}, {"mode", "frontend-bypass"}, {"supports", json::array({"chat"})}, {"compat", "provider-adapter-v1"}}
                    })},
                    {"defaultProvider", defaultProvider},
                    {"compat", json{{"shmProtocolVersion", kSharedMemoryProtocolVersion}, {"outsidesLinkMode", currentOutsidesLinkMode()}}}
                }}
            };
            respondJson(cb, out); }, {drogon::Get});

        drogon::app().registerHandler("/api/runtime/features", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            try {
                bool ok = true;
                json body = parseRequestBody(req, ok);
                if (!ok) return badRequest(cb, "invalid json");
                auto result = applyRuntimeFeaturePatch(body);
                respondJson(cb, json{{"ok", true}, {"result", result}, {"features", getRuntimeFeatureState()}});
            } catch (const std::exception &e) {
                respondJson(cb, json{{"ok", false}, {"error", e.what()}}, drogon::k500InternalServerError);
            } catch (...) {
                respondJson(cb, json{{"ok", false}, {"error", "internal error"}}, drogon::k500InternalServerError);
            } }, {drogon::Patch});

        drogon::app().registerHandler("/api/addons", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            auto controller = pool_->getActive();
            if (!controller) return badRequest(cb, "controller not available");
            auto runtime = controller->runtime();
            auto local = std::dynamic_pointer_cast<RuntimeState>(runtime);
            if (!local) return badRequest(cb, "runtime not supported");
            respondJson(cb, json{{"ok", true}, {"addons", local->listAddons()}}); }, {drogon::Get});

        drogon::app().registerHandler("/api/addons/add", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            bool ok = true;
            json body = parseRequestBody(req, ok);
            if (!ok) return badRequest(cb, "invalid json");
            auto controller = pool_->getActive();
            if (!controller) return badRequest(cb, "controller not available");
            auto runtime = controller->runtime();
            auto local = std::dynamic_pointer_cast<RuntimeState>(runtime);
            if (!local) return badRequest(cb, "runtime not supported");
            std::string path = body.value("path", body.value("libraryPath", ""));
            std::string type = body.value("type", "");
            std::string name = body.value("name", "");
            std::string error;
            bool okAdd = false;
            if (!path.empty()) {
                okAdd = local->addAddon(path, error);
            } else {
                okAdd = local->addAddon(type, name, error);
            }
            if (!okAdd) {
                respondJson(cb, json{{"ok", false}, {"error", error.empty() ? "addon add failed" : error}}, drogon::k400BadRequest);
                return;
            }
            respondJson(cb, json{{"ok", true}, {"addons", local->listAddons()}}); }, {drogon::Post});

        drogon::app().registerHandler("/api/addons/remove", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            bool ok = true;
            json body = parseRequestBody(req, ok);
            if (!ok) return badRequest(cb, "invalid json");
            auto controller = pool_->getActive();
            if (!controller) return badRequest(cb, "controller not available");
            auto runtime = controller->runtime();
            auto local = std::dynamic_pointer_cast<RuntimeState>(runtime);
            if (!local) return badRequest(cb, "runtime not supported");
            std::string name = body.value("name", "");
            std::string error;
            if (!local->removeAddon(name, error)) {
                respondJson(cb, json{{"ok", false}, {"error", error.empty() ? "addon remove failed" : error}}, drogon::k400BadRequest);
                return;
            }
            respondJson(cb, json{{"ok", true}, {"addons", local->listAddons()}}); }, {drogon::Post});

        drogon::app().registerHandler("/api/study/status", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            if (!study_) {
                respondJson(cb, json{{"ok", true}, {"running", false}, {"queue", 0}, {"metrics", nullptr}});
                return;
            }
            auto status = study_->status();
            json metrics = status.contains("metrics") ? status["metrics"] : json();
            respondJson(cb, json{{"ok", true}, {"running", status.value("running", false)}, {"queue", status.value("queue", 0)}, {"metrics", metrics}}); }, {drogon::Get});

        drogon::app().registerHandler("/api/monitoring/stats", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            respondJson(cb, json{{"ok", true}, {"result", getMonitoringStats()}}); }, {drogon::Get});

        drogon::app().registerHandler("/api/monitoring/training", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            respondJson(cb, json{{"ok", true}, {"result", getTrainMonitorStats()}}); }, {drogon::Get});

        drogon::app().registerHandler("/api/monitoring/reset", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            resetMonitoringStats();
            respondJson(cb, json{{"ok", true}, {"result", getMonitoringStats()}}); }, {drogon::Post});

        drogon::app().registerHandler("/api/monitoring/training/reset", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            resetTrainMonitor();
            respondJson(cb, json{{"ok", true}, {"result", getTrainMonitorStats()}}); }, {drogon::Post});

        drogon::app().registerHandler("/api/learn/dialog/reset", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            dialogCounters_ = {};
            respondJson(cb, json{{"ok", true}, {"dialogCounters", json{{"total", dialogCounters_.total}, {"lastRL", dialogCounters_.lastRL}, {"lastADV", dialogCounters_.lastADV}, {"lastGNN", dialogCounters_.lastGNN}}}}); }, {drogon::Post});

        drogon::app().registerHandler("/api/array/chat", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            try {
                bool ok = true;
                json body = parseRequestBody(req, ok);
                if (!ok) return badRequest(cb, "invalid json");
                auto payload = std::make_shared<json>(body);
                if (!payload || !payload->contains("text")) return badRequest(cb, "text required");
                auto &textVal = (*payload)["text"];
                bool truthy = true;
                if (textVal.is_null()) truthy = false;
                else if (textVal.is_boolean()) truthy = textVal.get<bool>();
                else if (textVal.is_number()) truthy = (textVal.get<double>() != 0.0);
                else if (textVal.is_string()) truthy = !textVal.get<std::string>().empty();
                if (!truthy) return badRequest(cb, "text required");
                json opts = payload->value("options", json::object());
                auto jsNumber = [&](const json &v, double fallback) -> double {
                    if (v.is_number()) return v.get<double>();
                    if (v.is_boolean()) return v.get<bool>() ? 1.0 : 0.0;
                    if (v.is_string()) {
                        double out = 0.0;
                        if (parseNumberLikeJs(v.get<std::string>(), out)) return out;
                        return fallback;
                    }
                    return fallback;
                };
                auto jsBool = [&](const json &v, bool fallback) -> bool {
                    if (v.is_boolean()) return v.get<bool>();
                    if (v.is_number_integer()) return v.get<int>() != 0;
                    if (v.is_number_float()) return std::fabs(v.get<double>()) > 1e-12;
                    if (v.is_string()) {
                        std::string s = trimCopy(v.get<std::string>());
                        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
                        if (s == "1" || s == "true" || s == "yes" || s == "on") return true;
                        if (s == "0" || s == "false" || s == "no" || s == "off") return false;
                    }
                    return fallback;
                };
                std::string text = textVal.is_string() ? textVal.get<std::string>() : textVal.dump();
                (*payload)["text"] = text;
                int bigRounds = 1;
                if (opts.contains("bigRounds")) {
                    double v = jsNumber(opts["bigRounds"], 1.0);
                    if (std::isfinite(v)) {
                        bigRounds = (v == 0.0) ? 1 : (int)v;
                    }
                } else if (opts.contains("bigSparkRounds")) {
                    double v = jsNumber(opts["bigSparkRounds"], 1.0);
                    if (std::isfinite(v)) {
                        bigRounds = (v == 0.0) ? 1 : (int)v;
                    }
                }

                bool includeTransformerNlp = true;
                if (opts.contains("includeTransformerNlp")) {
                    includeTransformerNlp = jsBool(opts["includeTransformerNlp"], true);
                }
                if (includeTransformerNlp && config_.inferenceEnabled) {
                    std::string backendMode = trimCopy(config_.transformerBackend);
                    std::transform(backendMode.begin(), backendMode.end(), backendMode.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
                    if (backendMode.empty())
                        backendMode = config_.usingOllama ? (config_.ollamaFineTuning ? "ollama-fine-tuning" : "ollama") : "native";

                    bool modeOff = (backendMode == "off" || backendMode == "false" || backendMode == "disabled");
                    if (!modeOff) {
                        int maxTokens = 256;
                        if (payload->contains("maxTokens")) {
                            double v = jsNumber((*payload)["maxTokens"], 256.0);
                            if (std::isfinite(v) && v > 0) maxTokens = std::max(16, (int)v);
                        } else if (payload->contains("max_tokens")) {
                            double v = jsNumber((*payload)["max_tokens"], 256.0);
                            if (std::isfinite(v) && v > 0) maxTokens = std::max(16, (int)v);
                        } else if (opts.contains("maxTokens")) {
                            double v = jsNumber(opts["maxTokens"], 256.0);
                            if (std::isfinite(v) && v > 0) maxTokens = std::max(16, (int)v);
                        }

                        std::string graphContext = payload->value("graphContext", "");
                        json transformerReply = json::object();
                        bool usingOllama = (backendMode == "ollama" || backendMode == "ollama-fine-tuning");
                        bool usingLlamaCpp = (backendMode == "llamacpp" || backendMode == "llama.cpp" || backendMode == "llama_cpp");
                        bool usingBitNet = (backendMode == "bitnet");

                        try {
                            if (usingOllama) transformerReply = chatWithOllama(text, graphContext, maxTokens);
                            else if (usingLlamaCpp) transformerReply = chatWithLlamaCpp(text, graphContext, maxTokens);
                            else if (usingBitNet) transformerReply = chatWithBitNet(text, graphContext, maxTokens);
                            else {
                                std::lock_guard<std::mutex> lock(transformerMu_);
                                std::vector<std::vector<float>> graphEmb;
                                transformerReply = transformer_.chat(text, graphContext, graphEmb, maxTokens);
                            }
                        } catch (...) {
                            transformerReply = json::object();
                        }

                        std::string reply = transformerReply.value("reply", "");
                        if (!reply.empty()) {
                            auto reqEmb = textToMiniEmbedding(text, 64);
                            auto repEmb = textToMiniEmbedding(reply, 64);
                            json candidate = json{{"source", backendMode},
                                                  {"reply", reply},
                                                  {"latency", transformerReply.value("latency", 0.0)},
                                                  {"affinity", cosineSim(reqEmb, repEmb)}};
                            (*payload)["sparkTransformerCandidates"] = json::array({candidate});
                        }
                    }
                }

                json result = bigRounds > 1 ? spark_->dispatchBig(*payload, opts) : spark_->dispatch(*payload, opts);
                respondJson(cb, json{{"ok", true}, {"result", result}});
            } catch (const std::exception &e) {
                respondJson(cb, json{{"ok", false}, {"error", e.what()}}, drogon::k500InternalServerError);
            } catch (...) {
                respondJson(cb, json{{"ok", false}, {"error", "internal error"}}, drogon::k500InternalServerError);
            } }, {drogon::Post});

        drogon::app().registerHandler("/api/learn/reinforce", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            if (rlDisabled_) return respondJson(cb, json{{"ok", false}, {"error", "rl-disabled"}}, drogon::k503ServiceUnavailable);
            try {
                bool ok = true;
                json body = parseRequestBody(req, ok);
                if (!ok) return badRequest(cb, "invalid json");
                auto payload = std::make_shared<json>(body);
                int cycles = 3;
                if (payload && payload->contains("cycles")) {
                    auto jsNumber = [&](const json &v, double fallback) -> double {
                        if (v.is_number()) return v.get<double>();
                        if (v.is_boolean()) return v.get<bool>() ? 1.0 : 0.0;
                        if (v.is_string()) {
                            double out = 0.0;
                            if (parseNumberLikeJs(v.get<std::string>(), out)) return out;
                            return fallback;
                        }
                        return fallback;
                    };
                    double v = jsNumber((*payload)["cycles"], 3.0);
                    if (std::isfinite(v)) cycles = (v == 0.0) ? 3 : (int)v;
                }
                auto out = ensureRL()->learn(cycles);
                json bridge = bridgeLearnerResultToExternalBackend("reinforcement", config_.transformerBackend, body, out);
                respondJson(cb, json{{"ok", true}, {"result", out}, {"externalBridge", bridge}});
            } catch (const std::exception &e) {
                respondJson(cb, json{{"ok", false}, {"error", e.what()}}, drogon::k500InternalServerError);
            } catch (...) {
                respondJson(cb, json{{"ok", false}, {"error", "internal error"}}, drogon::k500InternalServerError);
            } }, {drogon::Post});

        drogon::app().registerHandler("/api/learn/reinforce/latest", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            respondJson(cb, json{{"ok", true}, {"latest", rl_ ? rl_->latest() : json()}}); }, {drogon::Get});

        drogon::app().registerHandler("/api/learn/gnn_ga", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            if (gnnGaDisabled_) return respondJson(cb, json{{"ok", false}, {"error", "gnn-ga-disabled"}}, drogon::k503ServiceUnavailable);
            try {
                bool ok = true;
                json body = parseRequestBody(req, ok);
                if (!ok) return badRequest(cb, "invalid json");
                int generations = body.value("generations", 3);
                int population = body.value("population", 8);
                double residualWeight = body.value("residualWeight", 0.35);
                double mutationRate = body.value("mutationRate", 0.25);
                double mutationScale = body.value("mutationScale", 0.35);
                std::vector<std::string> samples;
                if (body.contains("samples") && body["samples"].is_array()) {
                    for (auto &s : body["samples"]) if (s.is_string()) samples.push_back(s.get<std::string>());
                }
                auto out = ensureGnnGa()->evolve(generations, population, samples, residualWeight, mutationRate, mutationScale);
                json bridge = bridgeLearnerResultToExternalBackend("gnn-ga", config_.transformerBackend, body, out);
                respondJson(cb, json{{"ok", true}, {"result", out}, {"externalBridge", bridge}});
            } catch (const std::exception &e) {
                respondJson(cb, json{{"ok", false}, {"error", e.what()}}, drogon::k500InternalServerError);
            } catch (...) {
                respondJson(cb, json{{"ok", false}, {"error", "internal error"}}, drogon::k500InternalServerError);
            } }, {drogon::Post});

        drogon::app().registerHandler("/api/learn/gnn_ga/latest", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            respondJson(cb, json{{"ok", true}, {"latest", gnnGa_ ? gnnGa_->latest() : json()}}); }, {drogon::Get});

        drogon::app().registerHandler("/api/learn/adversarial", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            if (advDisabled_) return respondJson(cb, json{{"ok", false}, {"error", "adv-disabled"}}, drogon::k503ServiceUnavailable);
            try {
                bool ok = true;
                json body = parseRequestBody(req, ok);
                if (!ok) return badRequest(cb, "invalid json");
                auto payload = std::make_shared<json>(body);
                if (!payload || !payload->contains("samples") || !(*payload)["samples"].is_array()) return badRequest(cb, "samples required");
                if ((*payload)["samples"].empty()) return badRequest(cb, "samples required");
                auto out = ensureADV()->attackAndDefend((*payload)["samples"]);
                json bridge = bridgeLearnerResultToExternalBackend("adversarial", config_.transformerBackend, body, out);
                respondJson(cb, json{{"ok", true}, {"result", out}, {"externalBridge", bridge}});
            } catch (const std::exception &e) {
                respondJson(cb, json{{"ok", false}, {"error", e.what()}}, drogon::k500InternalServerError);
            } catch (...) {
                respondJson(cb, json{{"ok", false}, {"error", "internal error"}}, drogon::k500InternalServerError);
            } }, {drogon::Post});

        drogon::app().registerHandler("/api/learn/adversarial/latest", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            respondJson(cb, json{{"ok", true}, {"latest", adv_ ? adv_->latest() : json()}}); }, {drogon::Get});

        drogon::app().registerHandler("/api/learn/thresholds", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            bool ok = true;
            json body = parseRequestBody(req, ok);
            if (!ok) return badRequest(cb, "invalid json");
            if (body.is_object()) {
                if (body.contains("rlEvery") && body["rlEvery"].is_number()) {
                    double v = body["rlEvery"].get<double>();
                    if (std::isfinite(v) && v > 0) dialogThresholds_.rlEvery = v;
                }
                if (body.contains("advEvery") && body["advEvery"].is_number()) {
                    double v = body["advEvery"].get<double>();
                    if (std::isfinite(v) && v > 0) dialogThresholds_.advEvery = v;
                }
                if (body.contains("gnnEvery") && body["gnnEvery"].is_number()) {
                    double v = body["gnnEvery"].get<double>();
                    if (std::isfinite(v) && v > 0) dialogThresholds_.gnnEvery = v;
                }
            }
            respondJson(cb, json{{"ok", true}, {"thresholds", json{{"rlEvery", dialogThresholds_.rlEvery}, {"advEvery", dialogThresholds_.advEvery}, {"gnnEvery", dialogThresholds_.gnnEvery}}}}); }, {drogon::Post});

        drogon::app().registerHandler("/api/memebarrier/start", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            try {
                bool ok = true;
                json body = parseRequestBody(req, ok);
                if (!ok) return badRequest(cb, "invalid json");
                double th = MODEL_DEFAULTS.maliciousThreshold;
                if (body.contains("maliciousThreshold") && body["maliciousThreshold"].is_number()) {
                    th = body["maliciousThreshold"].get<double>();
                }
                auto rt = std::dynamic_pointer_cast<RuntimeState>(pool_->getActive()->runtime());
                if (!rt) return respondJson(cb, json{{"ok", false}, {"error", "memebarrier not supported on current backend"}}, drogon::k400BadRequest);
                if (!barrier_) barrier_ = std::make_shared<MemeBarrier>(rt, 10000, th, 5);
                barrier_->setThreshold(th);
                barrier_->start();
                respondJson(cb, json{{"ok", true}, {"running", true}, {"threshold", barrier_->threshold()}});
            } catch (const std::exception &e) {
                respondJson(cb, json{{"ok", false}, {"error", e.what()}}, drogon::k500InternalServerError);
            } catch (...) {
                respondJson(cb, json{{"ok", false}, {"error", "internal error"}}, drogon::k500InternalServerError);
            } }, {drogon::Post});

        drogon::app().registerHandler("/api/memebarrier/stop", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            try {
                if (barrier_) barrier_->stop();
                respondJson(cb, json{{"ok", true}, {"running", false}});
            } catch (const std::exception &e) {
                respondJson(cb, json{{"ok", false}, {"error", e.what()}}, drogon::k500InternalServerError);
            } catch (...) {
                respondJson(cb, json{{"ok", false}, {"error", "internal error"}}, drogon::k500InternalServerError);
            } }, {drogon::Post});

        drogon::app().registerHandler("/api/memebarrier/stats", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            try {
                json stats;
                if (barrier_) {
                    auto s = barrier_->getStats();
                    stats = json{{"scans", s.scans}, {"isolated", s.isolated}, {"lastScanTime", s.lastScanTime}, {"lastIsolated", s.lastIsolated}};
                }
                respondJson(cb, json{{"ok", true}, {"stats", stats}});
            } catch (const std::exception &e) {
                respondJson(cb, json{{"ok", false}, {"error", e.what()}}, drogon::k500InternalServerError);
            } catch (...) {
                respondJson(cb, json{{"ok", false}, {"error", "internal error"}}, drogon::k500InternalServerError);
            } }, {drogon::Get});
        drogon::app().registerHandler("/api/model/params", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            if (req->method() == drogon::Get) {
                respondJson(cb, json{{"ok", true}, {"params", pool_->getActive()->runtime()->cloneParams()}});
            } else if (req->method() == drogon::Post) {
                bool ok = true;
                json body = parseRequestBody(req, ok);
                if (!ok) return badRequest(cb, "invalid json");
                pool_->getActive()->applyParams(body);
                respondJson(cb, json{{"ok", true}, {"params", pool_->getActive()->runtime()->cloneParams()}});
            } }, {drogon::Get, drogon::Post});

        drogon::app().registerHandler("/api/model/params/reset", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            json defaults = json{
                {"decayFactor", MODEL_DEFAULTS.decayFactor},
                {"maxMemeWords", MODEL_DEFAULTS.maxMemeWords},
                {"minOverlapThreshold", MODEL_DEFAULTS.minOverlapThreshold},
                {"memeNgramMin", MODEL_DEFAULTS.memeNgramMin},
                {"memeNgramMax", MODEL_DEFAULTS.memeNgramMax},
                {"maliciousThreshold", MODEL_DEFAULTS.maliciousThreshold},
                {"learningIterations", MODEL_DEFAULTS.learningIterations},
                {"iteration", MODEL_DEFAULTS.iteration},
                {"threshold", MODEL_DEFAULTS.threshold},
                {"decay", MODEL_DEFAULTS.decay},
                {"decayK", MODEL_DEFAULTS.decayK},
                {"maxLen", MODEL_DEFAULTS.maxLen},
                {"edgeWeight", MODEL_DEFAULTS.edgeWeight},
                {"activationType", MODEL_DEFAULTS.activationType},
                {"transferType", MODEL_DEFAULTS.transferType},
                {"activationCustom", MODEL_DEFAULTS.activationCustom},
                {"transferCustom", MODEL_DEFAULTS.transferCustom},
                {"mappingDepth", MODEL_DEFAULTS.mappingDepth},
                {"reflectionTopMemes", MODEL_DEFAULTS.reflectionTopMemes},
                {"reflectionTopWords", MODEL_DEFAULTS.reflectionTopWords},
                {"reflectionMinScore", MODEL_DEFAULTS.reflectionMinScore}
            };
            pool_->getActive()->applyParams(defaults);
            respondJson(cb, json{{"ok", true}, {"params", pool_->getActive()->runtime()->cloneParams()}}); }, {drogon::Post});

        drogon::app().registerHandler("/api/array/layers", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            json history = json::array();
            auto &h = spark_->history();
            int start = (int)h.size() > 20 ? (int)h.size() - 20 : 0;
            for (int i = start; i < (int)h.size(); i++) history.push_back(h[i]);
            respondJson(cb, json{{"ok", true}, {"layers", spark_->getLayers()}, {"history", history}}); }, {drogon::Get});

        drogon::app().registerHandler("/api/array/layers", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            bool ok = true;
            json body = parseRequestBody(req, ok);
            if (!ok) return badRequest(cb, "invalid json");
            try {
                spark_->updateLayers(body.value("layers", json::array()));
                respondJson(cb, json{{"ok", true}, {"layers", spark_->getLayers()}});
            } catch (const std::exception &e) {
                respondJson(cb, json{{"ok", false}, {"error", e.what()}}, drogon::k400BadRequest);
            } }, {drogon::Post});

        drogon::app().registerHandler("/api/array/history", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            json history = json::array();
            auto &h = spark_->history();
            int start = (int)h.size() > 20 ? (int)h.size() - 20 : 0;
            for (int i = start; i < (int)h.size(); i++) history.push_back(h[i]);
            respondJson(cb, json{{"ok", true}, {"history", history}}); }, {drogon::Get});

        drogon::app().registerHandler("/api/system/status", [this](const drogon::HttpRequestPtr &, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            json out;
            const auto now = std::chrono::steady_clock::now();
            const auto uptime = std::chrono::duration_cast<std::chrono::duration<double>>(now - startedAt_).count();
            out["ok"] = true;
            out["uptime"] = uptime;
            out["load"] = getLoadAvg();
            out["processCpuPercent"] = getProcessCpuPercent();
            out["memory"] = getMemoryUsage();
            out["controllers"] = pool_->listMetrics();
            out["groups"] = json::array();
            for (auto &gid : pool_->listGroupIds()) {
                out["groups"].push_back({{"gid", gid}, {"controllers", pool_->listControllersInGroup(gid)}});
            }
            out["rotation"] = json{{"running", rotation_ ? rotation_->running() : false}, {"cycleMs", rotation_ ? rotation_->cycleMs() : 0}};
            out["redis"] = json{{"channel", redis_ ? redis_->channel() : std::string("" )}};
            respondJson(cb, out); }, {drogon::Get});

        drogon::app().registerHandler("/api/system/config", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            json cfg;
            cfg["groupCount"] = config_.groupCount;
            cfg["groupSize"] = config_.groupSize;
            cfg["sparkNumAI"] = config_.sparkNumAI;
            cfg["sparkBudget"] = config_.sparkBudget;
            cfg["groupIds"] = pool_->listGroupIds();
            cfg["gatewayHost"] = config_.gatewayHost;
            cfg["portGateway"] = config_.portGateway;
            cfg["portStudy"] = config_.portStudy;
            respondJson(cb, json{{"ok", true}, {"config", cfg}}); }, {drogon::Get});

        drogon::app().registerHandler("/api/groups", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            json out;
            out["ok"] = true;
            out["groupSize"] = config_.groupSize;
            out["groupCount"] = config_.groupCount;
            out["groups"] = json::array();
            for (auto &gid : pool_->listGroupIds()) {
                out["groups"].push_back({{"gid", gid}, {"controllers", pool_->listControllersInGroup(gid)}});
            }
            respondJson(cb, out); }, {drogon::Get});

        drogon::app().registerHandler("/api/groups/{1}/metrics", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb, const std::string &gid)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            auto list = pool_->listControllersInGroup(gid);
            if (list.empty()) return respondJson(cb, json{{"ok", false}, {"error", "group-not-found"}}, drogon::k404NotFound);
            json metrics = json::array();
            if (isGroupProc()) {
                for (auto &name : list) {
                    (void)name;
                    metrics.push_back(json::object());
                }
            } else {
                for (auto &name : list) {
                    auto ctrl = pool_->getByName(name);
                    if (ctrl) metrics.push_back(ctrl->metrics());
                }
            }
            respondJson(cb, json{{"ok", true}, {"gid", gid}, {"controllers", metrics}}); }, {drogon::Get});

        drogon::app().registerHandler("/api/system/ai/{1}", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb, const std::string &name)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            auto ctrl = pool_->getByName(name);
            if (!ctrl) return respondJson(cb, json{{"ok", false}, {"error", "Controller not found"}}, drogon::k404NotFound);
            if (isGroupProc()) {
                respondJson(cb, json{{"ok", true}, {"controller", json::object()}});
            } else {
                respondJson(cb, json{{"ok", true}, {"controller", ctrl->metrics()}});
            } }, {drogon::Get});

        drogon::app().registerHandler("/api/shards", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            respondJson(cb, json{{"ok", true}, {"shards", shards_->metrics()}}); }, {drogon::Get});

        drogon::app().registerHandler("/api/study/enqueue", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            bool ok = true;
            json body = parseRequestBody(req, ok);
            if (!ok) return badRequest(cb, "invalid json");
            if (!body.contains("text") || !jsTruthy(body["text"])) return badRequest(cb, "text required");
            json doc{{"text", body["text"]}, {"queuedAt", nowEpochMs()}};
            if (study_) study_->enqueueDocument(doc);
            respondJson(cb, json{{"ok", true}}); }, {drogon::Post});

        drogon::app().registerHandler("/api/corpus/ingest", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            bool ok = true;
            json payload = parseRequestBody(req, ok);
            if (!ok) return badRequest(cb, "invalid json");
            if (!payload.contains("text") || !jsTruthy(payload["text"])) return badRequest(cb, "text required");
            try {
                auto results = pool_->ingestDocument(payload);
                if (payload.value("enqueueStudy", false) && study_) {
                    study_->enqueueDocument(payload);
                }
                respondJson(cb, json{{"ok", true}, {"results", results}});
            } catch (const std::exception &e) {
                respondJson(cb, json{{"ok", false}, {"error", e.what()}}, drogon::k500InternalServerError);
            } }, {drogon::Post});

        drogon::app().registerHandler("/api/corpus/forget", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            try {
                bool ok = true;
                json payload = parseRequestBody(req, ok);
                if (!ok) return badRequest(cb, "invalid json");
                auto results = pool_->forgetMemes(payload);
                respondJson(cb, json{{"ok", true}, {"results", results}});
            } catch (const std::exception &e) {
                respondJson(cb, json{{"ok", false}, {"error", e.what()}}, drogon::k500InternalServerError);
            } catch (...) {
                respondJson(cb, json{{"ok", false}, {"error", "internal error"}}, drogon::k500InternalServerError);
            } }, {drogon::Post});

        drogon::app().registerHandler("/api/corpus/online", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            try {
                bool ok = true;
                json payload = parseRequestBody(req, ok);
                if (!ok) return badRequest(cb, "invalid json");
                std::string query = payload.value("query", "");
                json tokens = payload.value("tokens", json::array());
                if (query.empty() && (!tokens.is_array() || tokens.empty())) return badRequest(cb, "query or tokens required");
                auto result = pool_->onlineResearch(!query.empty() ? json(query) : tokens, payload.value("options", json::object()));
                respondJson(cb, json{{"ok", true}, {"result", result}});
            } catch (const std::exception &e) {
                respondJson(cb, json{{"ok", false}, {"error", e.what()}}, drogon::k500InternalServerError);
            } catch (...) {
                respondJson(cb, json{{"ok", false}, {"error", "internal error"}}, drogon::k500InternalServerError);
            } }, {drogon::Post});

        drogon::app().registerHandler("/api/corpus/crawl", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            bool ok = true;
            json payload = parseRequestBody(req, ok);
            if (!ok) return badRequest(cb, "invalid json");
            std::string url = trimString(payload.value("startUrl", ""));
            if (url.empty()) return badRequest(cb, "startUrl required");
            json options = payload.value("options", json::object());
            json crawl = json{{"startUrl", url}};
            if (options.is_object()) {
                for (auto it = options.begin(); it != options.end(); ++it) {
                    crawl[it.key()] = it.value();
                }
            }
            json result = pool_->getActive()->runtime()->onlineLookup(url, json{{"mode", "crawl"}, {"crawl", crawl}});
            json ingested = nullptr;
            if (payload.value("ingest", false)) {
                json details = json::array();
                int docsCount = 0;
                for (auto &p : result.value("pages", json::array())) {
                    std::string text = p.value("text", "");
                    if (text.empty()) continue;
                    std::string source = payload.value("source", p.value("url", "crawl"));
                    json doc{{"text", text}, {"source", source}};
                    std::string gid = payload.value("groupId", "");
                    json r = gid.empty() ? pool_->ingestDocument(doc) : pool_->ingestDocumentToGroup(gid, doc);
                    details.push_back(json{{"source", source}, {"result", r}});
                    docsCount++;
                    if (docsCount >= 20) break;
                }
                ingested = json{{"docs", docsCount}, {"details", details}};
            }
            respondJson(cb, json{{"ok", true}, {"result", result}, {"ingested", ingested}}); }, {drogon::Post});

        drogon::app().registerHandler("/api/search/config", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            try {
                auto runtime = pool_->getActive()->runtime();
                respondJson(cb, json{{"ok", true}, {"config", runtime->getSearchConfig()}});
            } catch (const std::exception &e) {
                respondJson(cb, json{{"ok", false}, {"error", e.what()}}, drogon::k500InternalServerError);
            } catch (...) {
                respondJson(cb, json{{"ok", false}, {"error", "internal error"}}, drogon::k500InternalServerError);
            } }, {drogon::Get});

        drogon::app().registerHandler("/api/search/config", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            try {
                bool ok = true;
                json payload = parseRequestBody(req, ok);
                if (!ok) return badRequest(cb, "invalid json");
                auto runtime = pool_->getActive()->runtime();
                auto next = runtime->setSearchConfig(payload);
                respondJson(cb, json{{"ok", true}, {"config", next}});
            } catch (const std::exception &e) {
                respondJson(cb, json{{"ok", false}, {"error", e.what()}}, drogon::k500InternalServerError);
            } catch (...) {
                respondJson(cb, json{{"ok", false}, {"error", "internal error"}}, drogon::k500InternalServerError);
            } }, {drogon::Put});

        drogon::app().registerHandler("/api/search/endpoints/add", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            try {
                bool ok = true;
                json payload = parseRequestBody(req, ok);
                if (!ok) return badRequest(cb, "invalid json");
                std::string url = trimString(payload.value("url", ""));
                if (url.empty()) return badRequest(cb, "url required");
                auto runtime = pool_->getActive()->runtime();
                auto cfg = runtime->getSearchConfig();
                std::unordered_set<std::string> uniq;
                json endpoints = json::array();
                for (auto &e : cfg.value("endpoints", json::array())) {
                    if (e.is_string() && uniq.insert(e.get<std::string>()).second) endpoints.push_back(e);
                }
                if (uniq.insert(url).second) endpoints.push_back(url);
                auto next = runtime->setSearchConfig(json{{"endpoints", endpoints}});
                respondJson(cb, json{{"ok", true}, {"config", next}});
            } catch (const std::exception &e) {
                respondJson(cb, json{{"ok", false}, {"error", e.what()}}, drogon::k500InternalServerError);
            } catch (...) {
                respondJson(cb, json{{"ok", false}, {"error", "internal error"}}, drogon::k500InternalServerError);
            } }, {drogon::Post});

        drogon::app().registerHandler("/api/search/endpoints/remove", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            try {
                bool ok = true;
                json payload = parseRequestBody(req, ok);
                if (!ok) return badRequest(cb, "invalid json");
                std::string url = trimString(payload.value("url", ""));
                if (url.empty()) return badRequest(cb, "url required");
                auto runtime = pool_->getActive()->runtime();
                auto cfg = runtime->getSearchConfig();
                json endpoints = json::array();
                for (auto &e : cfg.value("endpoints", json::array())) {
                    if (e.is_string() && e.get<std::string>() != url) endpoints.push_back(e);
                }
                json patch{{"endpoints", endpoints}};
                if (cfg.value("active", "") == url) patch["active"] = endpoints.empty() ? "" : endpoints[0];
                auto next = runtime->setSearchConfig(patch);
                respondJson(cb, json{{"ok", true}, {"config", next}});
            } catch (const std::exception &e) {
                respondJson(cb, json{{"ok", false}, {"error", e.what()}}, drogon::k500InternalServerError);
            } catch (...) {
                respondJson(cb, json{{"ok", false}, {"error", "internal error"}}, drogon::k500InternalServerError);
            } }, {drogon::Post});

        drogon::app().registerHandler("/api/export/graph", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            try {
                bool ok = true;
                json body = parseRequestBody(req, ok);
                if (!ok) return badRequest(cb, "invalid json");
                json opts{{"seeds", body.value("seeds", json())}, {"radius", body.value("radius", json())}, {"file", body.value("file", json())}};
                auto runtime = pool_->getActive()->runtime();
                auto outFile = runtime->exportGraphToFile(opts);
                auto content = readTextFile(outFile);
                respondJson(cb, json{{"ok", true}, {"file", outFile.filename().string()}, {"path", outFile.string()}, {"content", content}});
            } catch (const std::exception &e) {
                respondJson(cb, json{{"ok", false}, {"error", e.what()}}, drogon::k500InternalServerError);
            } catch (...) {
                respondJson(cb, json{{"ok", false}, {"error", "internal error"}}, drogon::k500InternalServerError);
            } }, {drogon::Post});

        drogon::app().registerHandler("/api/export/graph/group", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            try {
                bool ok = true;
                json payload = parseRequestBody(req, ok);
                if (!ok) return badRequest(cb, "invalid json");
                std::string gid = trimString(payload.value("groupId", ""));
                if (gid.empty()) return badRequest(cb, "groupId required");
                auto list = pool_->listControllersInGroup(gid);
                if (list.empty()) return respondJson(cb, json{{"ok", false}, {"error", "group-not-found"}}, drogon::k404NotFound);
                auto ctrl = pool_->getByName(list.front());
                if (!ctrl) return respondJson(cb, json{{"ok", false}, {"error", "controller-not-found"}}, drogon::k404NotFound);
                json opts{{"seeds", payload.value("seeds", json())}, {"radius", payload.value("radius", json())}, {"file", payload.value("file", json())}};
                auto outFile = ctrl->runtime()->exportGraphToFile(opts);
                auto content = readTextFile(outFile);
                respondJson(cb, json{{"ok", true}, {"groupId", gid}, {"file", outFile.filename().string()}, {"path", outFile.string()}, {"content", content}});
            } catch (const std::exception &e) {
                respondJson(cb, json{{"ok", false}, {"error", e.what()}}, drogon::k500InternalServerError);
            } catch (...) {
                respondJson(cb, json{{"ok", false}, {"error", "internal error"}}, drogon::k500InternalServerError);
            } }, {drogon::Post});

        drogon::app().registerHandler("/api/robots/retrain", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            auto routeStarted = std::chrono::steady_clock::now();
            auto finalizeMetric = [&](bool okFlag) {
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - routeStarted).count();
                recordRouteMetric("/api/robots/retrain", elapsed, okFlag);
            };
            if (!authOK(req)) {
                finalizeMetric(false);
                return unauthorized(cb);
            }
            try {
                bool ok = true;
                json options = parseRequestBody(req, ok);
                if (!ok) {
                    finalizeMetric(false);
                    return badRequest(cb, "invalid json");
                }
                json files = json();
                if (options.contains("files")) {
                    if (options["files"].is_array()) {
                        files = options["files"];
                    } else if (options["files"].is_string()) {
                        std::string f = trimString(options["files"].get<std::string>());
                        if (!f.empty()) files = json::array({f});
                    }
                }
                bool shuffle = options.contains("shuffle") ? jsTruthy(options["shuffle"]) : false;
                int64_t limit = 2000;
                int64_t offset = 0;
                int64_t limitCap = getRobotsRetrainLimitCap();
                int64_t batchSize = getRobotsRetrainBatchDefault();
                int64_t batchCharsBudget = getRobotsRetrainBatchCharsBudget();
                int64_t docMaxChars = getRobotsRetrainDocMaxChars();
                int64_t docMaxTokens = getRobotsRetrainDocMaxTokens();
                std::string parseErr;
                if (!parseBoundedInt64(options, "limit", 1, limitCap, limit, parseErr, false)) {
                    finalizeMetric(false);
                    return badRequest(cb, parseErr);
                }
                if (!parseBoundedInt64(options, "offset", 0, 1000000000LL, offset, parseErr, false)) {
                    finalizeMetric(false);
                    return badRequest(cb, parseErr);
                }
                if (!parseBoundedInt64(options, "batchSize", 64, 5000, batchSize, parseErr, false)) {
                    finalizeMetric(false);
                    return badRequest(cb, parseErr);
                }
                batchSize = std::min(batchSize, limit);
                if (retrainTrace_) {
                    appendFatalLog(std::string("retrain-start limit=") + std::to_string(limit) +
                                   " offset=" + std::to_string(offset) +
                                   " batch=" + std::to_string(batchSize));
                }
                uint64_t monitorId = beginTrainMonitor("/api/robots/retrain", json{{"limit", limit}, {"offset", offset}, {"shuffle", shuffle}, {"enqueueStudy", options.value("enqueueStudy", false)}});
                touchTrainMonitor("collect-documents", 0, 0, 0, 0);
                auto runtime = pool_->getActive()->runtime();
                int64_t remaining = limit;
                int64_t cursorOffset = offset;
                int64_t collected = 0;
                int64_t batchIndex = 0;
                int64_t ingested = 0;
                int64_t memes = 0;
                int64_t edges = 0;
                int64_t failed = 0;
                uint64_t totalChars = 0;
                double avgDocChars = 1200.0;
                bool anyDocs = false;
                auto groups = pool_->listGroupIds();
                while (remaining > 0) {
                    batchIndex++;
                    int64_t take = std::min(batchSize, remaining);
                    if (batchCharsBudget > 0) {
                        double safeAvg = std::max(64.0, avgDocChars);
                        int64_t byChars = (int64_t)std::floor((double)batchCharsBudget / safeAvg);
                        byChars = std::max<int64_t>(32, byChars);
                        take = std::max<int64_t>(1, std::min<int64_t>(take, byChars));
                    }
                    json collect = json::object();
                    collect["limit"] = take;
                    collect["offset"] = cursorOffset;
                    collect["shuffle"] = shuffle;
                    if (files.is_array() && !files.empty()) collect["files"] = files;

                    auto docs = runtime->collectRobotsDocuments(collect);
                    if (docs.empty()) {
                        break;
                    }
                    uint64_t batchChars = 0;
                    for (const auto &doc : docs)
                        batchChars += (uint64_t)doc.text.size();
                    double batchAvg = docs.empty() ? avgDocChars : ((double)batchChars / (double)docs.size());
                    if (std::isfinite(batchAvg) && batchAvg > 0.0)
                        avgDocChars = avgDocChars * 0.75 + batchAvg * 0.25;
                    anyDocs = true;
                    collected += (int64_t)docs.size();
                    touchTrainMonitor("collect-documents", (uint64_t)limit, (uint64_t)collected, 0, totalChars);

                    for (auto &doc : docs) {
                        std::string docText = doc.text;
                        if ((int64_t)docText.size() > docMaxChars)
                            docText.resize((size_t)docMaxChars);
                        std::vector<std::string> docTokens = doc.tokens;
                        if ((int64_t)docTokens.size() > docMaxTokens)
                            docTokens.resize((size_t)docMaxTokens);
                        if (docText.empty() && docTokens.empty())
                            continue;

                        totalChars += (uint64_t)docText.size();
                        json j{{"text", docText}, {"tokens", docTokens}, {"source", doc.source}, {"file", doc.file}, {"id", doc.id}};
                        std::string sourceKey = !doc.source.empty() ? doc.source : (!doc.file.empty() ? doc.file : doc.id);
                        std::string key = sourceKey + "|" + docText.substr(0, 256);
                        int gi = groups.empty() ? 0 : (int)(hashStrSimple(key) % groups.size());
                        std::string target = groups.empty() ? "" : groups[gi];
                        json result = target.empty() ? pool_->ingestDocument(j) : pool_->ingestDocumentToGroup(target, j);
                        ingested++;
                        if (result.is_array()) {
                            for (auto &item : result) {
                                bool okItem = item.is_object() ? item.value("ok", true) : jsTruthy(item);
                                if (okItem && item.is_object()) {
                                    memes += item.value("memes", 0);
                                    edges += item.value("edges", 0);
                                } else if (!okItem) {
                                    failed++;
                                }
                            }
                        } else {
                            bool okItem = result.is_object() ? result.value("ok", true) : jsTruthy(result);
                            if (okItem && result.is_object()) {
                                memes += result.value("memes", 0);
                                edges += result.value("edges", 0);
                            } else if (!okItem) {
                                failed++;
                            }
                        }
                        if (options.value("enqueueStudy", false) && study_) study_->enqueueDocument(j);
                        if ((ingested % 64) == 0) {
                            touchTrainMonitor("ingest-progress", (uint64_t)limit, (uint64_t)ingested, 0, totalChars);
                        }
                    }

                    cursorOffset += (int64_t)docs.size();
                    remaining -= (int64_t)docs.size();
                    if (retrainTrace_ && (batchIndex % 10 == 0)) {
                        appendFatalLog(std::string("retrain-progress batch=") + std::to_string(batchIndex) +
                                       " take=" + std::to_string(take) +
                                       " avgDocChars=" + std::to_string((int64_t)avgDocChars) +
                                       " collected=" + std::to_string(collected) +
                                       " ingested=" + std::to_string(ingested) +
                                       " remaining=" + std::to_string(remaining));
                    }
                    if ((int64_t)docs.size() < take) {
                        break;
                    }
                }

                if (!anyDocs) {
                    finishTrainMonitor(false, json{{"ok", false}, {"error", "no-documents"}}, "no-documents");
                    finalizeMetric(false);
                    return respondJson(cb, json{{"ok", false}, {"error", "no-documents"}, {"monitorId", monitorId}}, drogon::k404NotFound);
                }

                json out{{"ok", true},
                         {"ingested", ingested},
                         {"memes", memes},
                         {"edges", edges},
                         {"failedControllers", failed},
                         {"requestedLimit", limit},
                         {"limitCap", limitCap},
                         {"batchSize", batchSize},
                         {"monitorId", monitorId}};
                if (ingested < limit) {
                    out["warning"] = "source exhausted before requested limit";
                } else {
                    out["warning"] = "requested limit reached; use offset for next batch";
                }
                finishTrainMonitor(true, out);
                if (retrainTrace_) {
                    appendFatalLog(std::string("retrain-done ingested=") + std::to_string(ingested) +
                                   " failed=" + std::to_string(failed));
                }
                finalizeMetric(true);
                respondJson(cb, out);
            } catch (const std::bad_alloc &e) {
                finishTrainMonitor(false, json::object(), std::string("bad_alloc: ") + e.what());
                if (retrainTrace_) {
                    appendFatalLog(std::string("retrain-bad-alloc: ") + e.what());
                }
                logRouteError("/api/robots/retrain", std::string("bad_alloc: ") + e.what());
                finalizeMetric(false);
                respondJson(cb, json{{"ok", false}, {"error", "insufficient memory during retrain"}}, drogon::k500InternalServerError);
            } catch (const std::exception &e) {
                finishTrainMonitor(false, json::object(), e.what());
                if (retrainTrace_) {
                    appendFatalLog(std::string("retrain-exception: ") + e.what());
                }
                respondJson(cb, json{{"ok", false}, {"error", e.what()}}, drogon::k500InternalServerError);
                finalizeMetric(false);
            } catch (...) {
                finishTrainMonitor(false, json::object(), "internal error");
                if (retrainTrace_) {
                    appendFatalLog("retrain-exception: internal error");
                }
                respondJson(cb, json{{"ok", false}, {"error", "internal error"}}, drogon::k500InternalServerError);
                finalizeMetric(false);
            } }, {drogon::Post});

        drogon::app().registerHandler("/robots/list", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            try {
                auto runtime = pool_->getActive()->runtime();
                respondJson(cb, json{{"ok", true}, {"directory", runtime->config().robotsDir.string()}, {"files", runtime->listRobotsFiles()}});
            } catch (const std::exception &e) {
                respondJson(cb, json{{"ok", false}, {"error", e.what()}}, drogon::k500InternalServerError);
            } catch (...) {
                respondJson(cb, json{{"ok", false}, {"error", "internal error"}}, drogon::k500InternalServerError);
            } }, {drogon::Get});

        drogon::app().registerHandler("/robots/ingest", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            try {
                bool ok = true;
                json options = parseRequestBody(req, ok);
                if (!ok) return badRequest(cb, "invalid json");
                json files = json();
                if (options.contains("files")) {
                    if (options["files"].is_array()) {
                        files = options["files"];
                    } else if (options["files"].is_string()) {
                        std::string f = trimString(options["files"].get<std::string>());
                        if (!f.empty()) files = json::array({f});
                    }
                }
                bool shuffle = options.contains("shuffle") ? jsTruthy(options["shuffle"]) : false;
                json collect = json::object();
                if (options.contains("limit")) collect["limit"] = options["limit"];
                if (options.contains("offset")) collect["offset"] = options["offset"];
                collect["shuffle"] = shuffle;
                if (files.is_array() && !files.empty()) collect["files"] = files;
                auto runtime = pool_->getActive()->runtime();
                auto docs = runtime->collectRobotsDocuments(collect);
                if (docs.empty()) return respondJson(cb, json{{"ok", false}, {"error", "no-documents"}}, drogon::k404NotFound);
                json summary = json::array();
                auto groups = pool_->listGroupIds();
                for (auto &doc : docs) {
                    json j{{"text", doc.text}, {"tokens", doc.tokens}, {"source", doc.source}, {"file", doc.file}, {"id", doc.id}};
                    std::string sourceKey = !doc.source.empty() ? doc.source : (!doc.file.empty() ? doc.file : doc.id);
                    std::string key = sourceKey + "|" + doc.text.substr(0, 256);
                    int gi = groups.empty() ? 0 : (int)(hashStrSimple(key) % groups.size());
                    std::string target = groups.empty() ? "" : groups[gi];
                    json result = target.empty() ? pool_->ingestDocument(j) : pool_->ingestDocumentToGroup(target, j);
                    int docMemes = 0;
                    int docEdges = 0;
                    int docFailed = 0;
                    if (result.is_array()) {
                        for (auto &item : result) {
                            bool okItem = item.is_object() ? item.value("ok", true) : jsTruthy(item);
                            if (okItem && item.is_object()) {
                                docMemes += item.value("memes", 0);
                                docEdges += item.value("edges", 0);
                            } else if (!okItem) {
                                docFailed++;
                            }
                        }
                    } else {
                        bool okItem = result.is_object() ? result.value("ok", true) : jsTruthy(result);
                        if (okItem && result.is_object()) {
                            docMemes += result.value("memes", 0);
                            docEdges += result.value("edges", 0);
                        } else if (!okItem) {
                            docFailed++;
                        }
                    }
                    json item{{"id", doc.id}, {"source", doc.source}, {"file", doc.file}, {"tokens", (int)doc.tokens.size()}, {"memes", docMemes}, {"edges", docEdges}, {"failedControllers", docFailed}};
                    summary.push_back(item);
                    if (options.value("enqueueStudy", false) && study_) study_->enqueueDocument(j);
                }
                std::unordered_set<std::string> seen;
                json filesOut = json::array();
                bool nullAdded = false;
                for (auto &item : summary) {
                    if (item.contains("file") && item["file"].is_string()) {
                        std::string f = item["file"].get<std::string>();
                        if (seen.insert(f).second) filesOut.push_back(f);
                    } else if (!nullAdded) {
                        filesOut.push_back(nullptr);
                        nullAdded = true;
                    }
                }
                respondJson(cb, json{{"ok", true}, {"ingested", (int)summary.size()}, {"files", filesOut}, {"summary", summary}});
            } catch (const std::exception &e) {
                respondJson(cb, json{{"ok", false}, {"error", e.what()}}, drogon::k500InternalServerError);
            } catch (...) {
                respondJson(cb, json{{"ok", false}, {"error", "internal error"}}, drogon::k500InternalServerError);
            } }, {drogon::Post});

        drogon::app().registerHandler("/api/tests/list", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            try {
                fs::path testsDir = pool_->getActive()->runtime()->config().testsDir;
                std::vector<std::string> names;
                if (fs::exists(testsDir)) {
                    for (auto &entry : fs::directory_iterator(testsDir)) {
                        if (!entry.is_regular_file()) continue;
                        auto ext = entry.path().extension().string();
                        std::string lowerExt;
                        lowerExt.reserve(ext.size());
                        for (char c : ext) lowerExt.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
                        if (lowerExt == ".txt") names.push_back(entry.path().filename().string());
                    }
                }
                std::sort(names.begin(), names.end());
                json files = json::array();
                for (auto &n : names) files.push_back(n);
                respondJson(cb, json{{"ok", true}, {"directory", testsDir.string()}, {"files", files}});
            } catch (const std::exception &e) {
                respondJson(cb, json{{"ok", false}, {"error", e.what()}}, drogon::k500InternalServerError);
            } catch (...) {
                respondJson(cb, json{{"ok", false}, {"error", "internal error"}}, drogon::k500InternalServerError);
            } }, {drogon::Get});

        drogon::app().registerHandler("/api/tests/case", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            try {
                bool ok = true;
                json payload = parseRequestBody(req, ok);
                if (!ok) return badRequest(cb, "invalid json");
                std::function<std::string(const json &)> jsStringForJoin;
                std::function<std::string(const json &)> jsStringValue;
                jsStringForJoin = [&](const json &v) -> std::string {
                    if (v.is_null()) return "";
                    if (v.is_string()) return v.get<std::string>();
                    if (v.is_boolean()) return v.get<bool>() ? "true" : "false";
                    if (v.is_number()) return v.dump();
                    if (v.is_array()) {
                        std::string out;
                        bool first = true;
                        for (auto &item : v) {
                            if (!first) out += ",";
                            first = false;
                            out += jsStringForJoin(item);
                        }
                        return out;
                    }
                    return "[object Object]";
                };
                jsStringValue = [&](const json &v) -> std::string {
                    if (v.is_null()) return "null";
                    if (v.is_string()) return v.get<std::string>();
                    if (v.is_boolean()) return v.get<bool>() ? "true" : "false";
                    if (v.is_number()) return v.dump();
                    if (v.is_array()) return jsStringForJoin(v);
                    return "[object Object]";
                };
                json nameVal = payload.contains("name") ? payload["name"] : json();
                json contentVal = payload.contains("content") ? payload["content"] : json();
                std::string nameRaw = trimString(jsTruthy(nameVal) ? jsStringValue(nameVal) : "");
                std::string content = jsTruthy(contentVal) ? jsStringValue(contentVal) : "";
                if (nameRaw.empty()) return badRequest(cb, "name required");
                std::string safe;
                safe.reserve(nameRaw.size());
                for (char c : nameRaw) {
                    if (std::isalnum(static_cast<unsigned char>(c)) || c == '.' || c == '_' || c == '-') safe.push_back(c);
                    else safe.push_back('_');
                }
                auto lower = safe;
                std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
                if (lower.size() < 4 || lower.substr(lower.size() - 4) != ".txt") safe += ".txt";
                fs::path testsDir = pool_->getActive()->runtime()->config().testsDir;
                ensureDir(testsDir);
                fs::path filePath = testsDir / safe;
                std::ofstream ofs(filePath, std::ios::binary);
                ofs << content;
                respondJson(cb, json{{"ok", true}, {"file", safe}, {"path", filePath.string()}});
            } catch (const std::exception &e) {
                respondJson(cb, json{{"ok", false}, {"error", e.what()}}, drogon::k500InternalServerError);
            } catch (...) {
                respondJson(cb, json{{"ok", false}, {"error", "internal error"}}, drogon::k500InternalServerError);
            } }, {drogon::Post});

        drogon::app().registerHandler("/api/tests/refresh", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            try {
                fs::path testsDir = pool_->getActive()->runtime()->config().testsDir;
                bool refreshed = false;
                if (rl_) {
                    try {
                        rl_->refreshTests(testsDir);
                        refreshed = true;
                    } catch (...) {
                        try {
                            rl_->setTestsDir(testsDir);
                            refreshed = true;
                        } catch (...) {}
                    }
                } else {
                    auto factory = module_mount::ModuleRegistry::instance().reinforcementLearnerFactory();
                    if (factory)
                        rl_ = factory(pool_, testsDir);
                    else
                        rl_ = std::make_shared<ReinforcementLearner>(pool_, testsDir);
                    refreshed = true;
                }
                respondJson(cb, json{{"ok", true}, {"refreshed", refreshed}, {"testsDir", testsDir.string()}});
            } catch (const std::exception &e) {
                respondJson(cb, json{{"ok", false}, {"error", e.what()}}, drogon::k500InternalServerError);
            } catch (...) {
                respondJson(cb, json{{"ok", false}, {"error", "internal error"}}, drogon::k500InternalServerError);
            } }, {drogon::Post});

        drogon::app().registerHandler("/api/snapshots", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            respondJson(cb, json{{"ok", true}, {"list", snapshots_->list()}}); }, {drogon::Get});

        drogon::app().registerHandler("/api/snapshots/create", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            bool ok = true;
            json payload = parseRequestBody(req, ok);
            if (!ok) return badRequest(cb, "invalid json");
            std::string name = payload.value("name", "manual");
            auto file = snapshots_->create(name);
            respondJson(cb, json{{"ok", true}, {"file", file.filename().string()}}); }, {drogon::Post});

        drogon::app().registerHandler("/api/snapshots/restore/{1}", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb, const std::string &id)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            try {
                snapshots_->restore(id);
                respondJson(cb, json{{"ok", true}});
            } catch (const std::exception &e) {
                respondJson(cb, json{{"ok", false}, {"error", e.what()}}, drogon::k404NotFound);
            } }, {drogon::Post});

        drogon::app().registerHandler("/api/snapshots/{1}", [this](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb, const std::string &id)
                                      {
            if (!authOK(req)) return unauthorized(cb);
            snapshots_->remove(id);
            respondJson(cb, json{{"ok", true}}); }, {drogon::Delete});
    }

    void onDialogCompleted(const json &result, const json &payload)
    {
        try
        {
            dialogCounters_.total += 1;
            if (logger_.enabled())
            {
                logger_.log(LoggerCXX::Type::LOG, buildDialogContextLog(result, payload));
                if (logger_.mode() == LoggerCXX::Mode::Debug)
                {
                    logger_.log(LoggerCXX::Type::MEMORY, buildMemorySnapshotLog());
                    logger_.log(LoggerCXX::Type::COMPUTE, buildComputeRouteLog(result));
                    logger_.log(LoggerCXX::Type::MONITORING, buildMonitoringLog());
                }
                else if (logger_.mode() == LoggerCXX::Mode::Release)
                {
                    if (!releaseSnapshotLogged_)
                    {
                        logger_.log(LoggerCXX::Type::MEMORY, buildMemorySnapshotLog());
                        releaseSnapshotLogged_ = true;
                    }
                }
            }
            if (!dialogLearningEnabled_)
                return;

            if (dialogAsyncInFlight_.load(std::memory_order_relaxed) >= std::max(1, dialogAsyncLimit_))
                return;

            std::string guardText = payload.value("text", payload.value("message", ""));
            int guardTokens = roughTokenCount(guardText);
            std::string guardReply = result.value("reply", "");
            bool guardLongQuery = result.value("longQuerySafeMode", false);
            std::transform(guardReply.begin(), guardReply.end(), guardReply.begin(), [](unsigned char c)
                           { return (char)std::tolower(c); });
            if (guardLongQuery || guardReply == "disconnected" || guardText.size() >= 180 || guardTokens >= 16)
                return;

            try
            {
                json wrapped = json{{"payload", payload}, {"result", result}};
                pool_->getActive()->runtime()->learnFromDialog(wrapped, json());
            }
            catch (...)
            {
            }

            try
            {
                std::string text = payload.value("text", payload.value("message", ""));
                std::string reply = result.value("reply", "");
                std::string graphContext = result.value("graphContext", "");
                if (!text.empty() && !reply.empty())
                {
                    bool longQuerySafeMode = result.value("longQuerySafeMode", false);
                    int textTokens = roughTokenCount(text);
                    std::string replyLower = reply;
                    std::transform(replyLower.begin(), replyLower.end(), replyLower.begin(), [](unsigned char c)
                                   { return (char)std::tolower(c); });
                    if (longQuerySafeMode || replyLower == "disconnected" || text.size() >= 180 || textTokens >= 16)
                        return;
                    if (isLikelyGibberishReply(reply))
                        return;
                    if (containsInternalContextMarkers(reply) || hasExcessUnknownTokens(reply))
                        return;

                    if (!dialogLearnGraphContext_)
                    {
                        graphContext.clear();
                    }
                    else
                    {
                        bool cleanedGraph = false;
                        graphContext = sanitizeInputText(graphContext, cleanedGraph);
                        if (containsInternalContextMarkers(graphContext))
                            graphContext.clear();
                        if (graphContext.size() > 512)
                            graphContext.resize(512);
                    }

                    float verifyScore = 0.0f;
                    if (result.contains("verify") && result["verify"].is_object())
                    {
                        verifyScore = result["verify"].value("score", 0.0f);
                    }
                    float addonSignal = 0.0f;
                    std::string feedbackSource = "implicit_dialog";
                    if (result.contains("graph") && result["graph"].is_object())
                    {
                        const auto &g = result["graph"];
                        if (g.contains("addon") && g["addon"].is_object())
                        {
                            addonSignal = 1.0f;
                            std::string n = g["addon"].value("name", g["addon"].value("addon", ""));
                            std::string t = g["addon"].value("type", "");
                            if (!n.empty())
                                feedbackSource = "addon:" + n;
                            else if (!t.empty())
                                feedbackSource = "addonType:" + t;
                        }
                    }
                    if (result.contains("sparkGnnStage2") && result["sparkGnnStage2"].is_string())
                    {
                        std::string s = result["sparkGnnStage2"].get<std::string>();
                        if (!s.empty())
                            addonSignal += 0.2f;
                    }
                    float addonScale = 0.35f;
                    {
                        std::lock_guard<std::mutex> lock(transformerMu_);
                        addonScale = transformer_.params().value("addonRewardScale", 0.35f);
                    }
                    float reward = std::max(-1.0f, std::min(1.0f, verifyScore + addonScale * addonSignal));
                    bool trainTick = dialogTransformerLearning_ && !longQuerySafeMode && (dialogCounters_.total % 6 == 0);
                    transformer::TrainSample sample{text, reply, graphContext};
                    if (dialogTransformerLearning_ && !longQuerySafeMode)
                    {
                        launchDialogTask([this, sample]()
                                         {
                            try {
                                std::unique_lock<std::mutex> lock(transformerMu_, std::try_to_lock);
                                if (!lock.owns_lock()) return;
                                transformer_.jointTrain(std::vector<transformer::TrainSample>{sample}, 1, 0.001f, 0.35f);
                            } catch (...) {
                            } });
                    }
                    if (dialogTransformerLearning_)
                    {
                        launchDialogTask([this, text, graphContext, reply, reward, trainTick, feedbackSource]()
                                         {
                            try {
                                std::unique_lock<std::mutex> lock(transformerMu_, std::try_to_lock);
                                if (!lock.owns_lock()) return;
                                transformer_.addPreferenceFeedback(text, graphContext, reply, std::string(), reward, feedbackSource);
                                if (trainTick)
                                {
                                    transformer_.trainFromFeedback(1);
                                }
                            } catch (...) {
                            } });
                    }
                }
            }
            catch (...)
            {
            }

            const auto total = dialogCounters_.total;
            if (!rlDisabled_ && (total - dialogCounters_.lastRL >= dialogThresholds_.rlEvery))
            {
                dialogCounters_.lastRL = total;
                launchDialogTask([this]()
                         {
                    try { ensureRL()->learn(1); } catch (...) {} });
            }
            if (!advDisabled_ && (total - dialogCounters_.lastADV >= dialogThresholds_.advEvery))
            {
                dialogCounters_.lastADV = total;
                std::string text;
                if (payload.contains("text") && payload["text"].is_string())
                {
                    text = payload["text"].get<std::string>();
                }
                else if (payload.contains("tokens") && payload["tokens"].is_array())
                {
                    std::ostringstream oss;
                    bool first = true;
                    for (auto &t : payload["tokens"])
                    {
                        if (!t.is_string())
                            continue;
                        if (!first)
                            oss << " ";
                        oss << t.get<std::string>();
                        first = false;
                    }
                    text = oss.str();
                }
                auto trim = [](std::string s)
                {
                    auto start = s.find_first_not_of(" \n\r\t");
                    if (start == std::string::npos)
                        return std::string();
                    auto end = s.find_last_not_of(" \n\r\t");
                    return s.substr(start, end - start + 1);
                };
                text = trim(text);
                std::vector<std::string> samples;
                if (!text.empty())
                    samples.push_back(text);
                if (result.contains("reply") && result["reply"].is_string())
                {
                    samples.push_back(result["reply"].get<std::string>());
                }
                if (!samples.empty())
                {
                    launchDialogTask([this, samples]()
                             {
                    try { ensureADV()->attackAndDefend(samples); } catch (...) {} });
                }
            }

            if (!gnnGaDisabled_ && (total - dialogCounters_.lastGNN >= dialogThresholds_.gnnEvery))
            {
                dialogCounters_.lastGNN = total;
                std::vector<std::string> samples;
                std::string text = payload.value("text", payload.value("message", ""));
                if (!text.empty())
                    samples.push_back(text);
                if (result.contains("reply") && result["reply"].is_string())
                {
                    std::string reply = result["reply"].get<std::string>();
                    if (!reply.empty())
                        samples.push_back(reply);
                }
                if (result.contains("graphContext") && result["graphContext"].is_string())
                {
                    std::string gc = result["graphContext"].get<std::string>();
                    if (dialogLearnGraphContext_ && !gc.empty() && !containsInternalContextMarkers(gc))
                        samples.push_back(gc);
                }
                launchDialogTask([this, samples]()
                         {
                    try { ensureGnnGa()->evolve(1, 6, samples, 0.35, 0.25, 0.30); } catch (...) {} });
            }
        }
        catch (...)
        {
        }
    }

    void respondJson(const std::function<void(const drogon::HttpResponsePtr &)> &cb, const json &j, drogon::HttpStatusCode code = drogon::k200OK)
    {
        try
        {
            if (logger_.enabled())
            {
                int sc = (int)code;
                std::string payload;
                try
                {
                    payload = j.dump();
                }
                catch (const std::exception &e)
                {
                    payload = std::string("<json-dump-failed: ") + e.what() + ">";
                }
                catch (...)
                {
                    payload = "<json-dump-failed: unknown>";
                }
                if (payload.size() > 2000)
                    payload = payload.substr(0, 2000) + "...";
                if (sc >= 500)
                {
                    logger_.log(LoggerCXX::Type::ERROR, std::string("http status=") + std::to_string(sc) + " payload=" + payload);
                }
                else if (sc >= 400)
                {
                    logger_.log(LoggerCXX::Type::WARNING, std::string("http status=") + std::to_string(sc) + " payload=" + payload);
                }
            }

            auto resp = drogon::HttpResponse::newHttpJsonResponse(toJsoncpp(j));
            resp->setStatusCode(code);
            cb(resp);
            return;
        }
        catch (const std::exception &e)
        {
            if (logger_.enabled())
                logger_.log(LoggerCXX::Type::ERROR, std::string("respondJson failed: ") + e.what());
            appendFatalLog(std::string("respondJson exception: ") + e.what());
        }
        catch (...)
        {
            if (logger_.enabled())
                logger_.log(LoggerCXX::Type::ERROR, "respondJson failed: unknown exception");
            appendFatalLog("respondJson unknown exception");
        }

        try
        {
            Json::Value fallback;
            fallback["ok"] = false;
            fallback["error"] = "response-encode-failed";
            auto resp = drogon::HttpResponse::newHttpJsonResponse(fallback);
            resp->setStatusCode(drogon::k500InternalServerError);
            cb(resp);
            return;
        }
        catch (...)
        {
        }

        auto resp = drogon::HttpResponse::newHttpResponse();
        resp->setStatusCode(drogon::k500InternalServerError);
        resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
        resp->setBody("{\"ok\":false,\"error\":\"response-failed\"}");
        cb(resp);
    }

    void unauthorized(const std::function<void(const drogon::HttpResponsePtr &)> &cb)
    {
        if (hasAuthError_)
        {
            json payload{{"ok", false}, {"error", lastAuthError_.error}};
            if (!lastAuthError_.message.empty())
                payload["message"] = lastAuthError_.message;
            logSecurityEvent(std::string("unauthorized response error=") + lastAuthError_.error);
            respondJson(cb, payload, lastAuthError_.code);
            return;
        }
        logSecurityEvent("unauthorized response error=unauthorized");
        respondJson(cb, json{{"ok", false}, {"error", "unauthorized"}}, drogon::k401Unauthorized);
    }

    void badRequest(const std::function<void(const drogon::HttpResponsePtr &)> &cb, const std::string &msg)
    {
        if (logger_.enabled())
            logger_.log(LoggerCXX::Type::WARNING, std::string("bad request: ") + msg);
        respondJson(cb, json{{"ok", false}, {"error", msg}}, drogon::k400BadRequest);
    }
};

thread_local GatewayServer::AuthError GatewayServer::lastAuthError_{};
thread_local bool GatewayServer::hasAuthError_ = false;

// runGroupWorker 启动组进程工作模式下的本地 RPC 服务。
// 调用方式：通过 group-worker 启动参数进入该函数。
// 实现思路：监听本地端口并按命令分发到组内运行时。
// 注意事项：端口与组配置缺失会直接返回错误码。
static int runGroupWorker(const Config &config)
{
    initProcessMemoryReservation();
    std::string groupId = trimString(getEnv("AI_GROUP_ID", "G1"));
    if (groupId.empty())
        groupId = "G1";
    int groupSize = std::max(1, (int)numberOr(getEnv("AI_GROUP_SIZE", "7"), 7));
    int port = 0;
    try
    {
        port = std::stoi(getEnv("AI_GROUP_WORKER_PORT", "0"));
    }
    catch (...)
    {
        port = 0;
    }
    if (port <= 0)
        return 2;

    fs::path lmdbRoot = getProjectRoot() / "lmdb";
    auto lmdbRootEnv = getEnv("AI_LMDB_ROOT", "");
    if (!lmdbRootEnv.empty())
        lmdbRoot = fs::path(lmdbRootEnv);

    std::size_t lmdbMapBytes = 512ull * 1024ull * 1024ull;
    auto raw = getEnv("AI_LMDB_MAP_BYTES", "");
    if (!raw.empty())
    {
        lmdbMapBytes = (std::size_t)numberOr(raw, (double)(512ull * 1024ull * 1024ull));
    }
    lmdbMapBytes = std::max<std::size_t>(64ull * 1024ull * 1024ull, lmdbMapBytes);

    ensureDir(config.baseDir);
    if (!config.dbPath.empty())
    {
        ensureDir(config.dbPath.parent_path());
    }

#ifdef _WIN32
    WSADATA wsaData{};
    WSAStartup(MAKEWORD(2, 2), &wsaData);
#endif

    SocketHandle serverSock = socket(AF_INET, SOCK_STREAM, 0);
    if (serverSock == kInvalidSocket)
        return 3;
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    int opt = 1;
#ifdef _WIN32
    setsockopt(serverSock, SOL_SOCKET, SO_REUSEADDR, (const char *)&opt, sizeof(opt));
#else
    setsockopt(serverSock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#endif
    if (bind(serverSock, (sockaddr *)&addr, sizeof(addr)) != 0)
        return 4;
    if (listen(serverSock, 1) != 0)
        return 5;

    SocketHandle client = accept(serverSock, nullptr, nullptr);
    if (client == kInvalidSocket)
        return 6;

    std::shared_ptr<KeyValueStore> kvmStore;
    std::shared_ptr<KeyValueStore> memeStore;
    std::shared_ptr<KeyValueStore> sessionStore;
#ifdef HAVE_SQLITE
    auto db = std::make_shared<Database079>(config.dbPath, lmdbRoot, config.redisUrl, config.redisCacheDb, config.redisCachePrefix);
    if (db->open())
    {
        kvmStore = db->createStore("kvm");
        memeStore = db->createStore("meme_graph");
        sessionStore = db->createStore("session");
    }
#endif
#ifdef HAVE_LMDB
    if (!Database079::isSqliteStore(kvmStore))
    {
        auto lmdbK = std::make_shared<LmdbStore>("kvm", lmdbRoot, lmdbMapBytes);
        auto lmdbM = std::make_shared<LmdbStore>("meme_graph", lmdbRoot, lmdbMapBytes);
        auto lmdbS = std::make_shared<LmdbStore>("session", lmdbRoot, lmdbMapBytes);
        if (lmdbK->ok())
            kvmStore = lmdbK;
        if (lmdbM->ok())
            memeStore = lmdbM;
        if (lmdbS->ok())
            sessionStore = lmdbS;
    }
#endif
    if (!kvmStore)
    {
        kvmStore = std::make_shared<JsonFileStore>("kvm", lmdbRoot);
        memeStore = std::make_shared<JsonFileStore>("meme_graph", lmdbRoot);
        sessionStore = std::make_shared<JsonFileStore>("session", lmdbRoot);
    }

    auto kvmNs = std::make_shared<NamespacedStore>(kvmStore, groupId + ":kvm");
    auto memeNs = std::make_shared<NamespacedStore>(memeStore, groupId + ":graph");
    auto sessNs = std::make_shared<NamespacedStore>(sessionStore, groupId + ":session");

    std::unordered_map<std::string, std::shared_ptr<RuntimeState>> runtimes;
    std::vector<std::string> controllerNames;
    for (int i = 0; i < groupSize; i++)
    {
        std::string name = groupId + "_AI" + std::to_string(i + 1);
        controllerNames.push_back(name);
        runtimes[name] = std::make_shared<RuntimeState>(kvmNs, memeNs, sessNs, config);
    }

    std::shared_ptr<MemeBarrier> memeBarrier;
    auto getRuntime = [&](const std::string &name) -> std::shared_ptr<RuntimeState>
    {
        auto it = runtimes.find(name);
        if (it == runtimes.end())
            throw std::runtime_error("controller-not-found:" + name);
        return it->second;
    };

    auto safeCall = [&](const std::function<json()> &fn) -> json
    {
        try
        {
            return json{{"ok", true}, {"result", fn()}};
        }
        catch (const std::exception &e)
        {
            return json{{"ok", false}, {"error", std::string(e.what())}};
        }
    };

    std::vector<uint8_t> requestBytes;
    requestBytes.reserve(8ull * 1024ull * 1024ull);
    while (recvPacket(client, requestBytes, 600000))
    {
        json msg;
        try
        {
            msg = json::from_msgpack(requestBytes);
        }
        catch (...)
        {
            continue;
        }
        std::string id = msg.value("id", "");
        std::string cmd = msg.value("cmd", "");
        if (id.empty() || cmd.empty())
            continue;

        int shmProtocol = msg.value("shmProtocolVersion", 1);
        if (shmProtocol > kSharedMemoryProtocolVersion)
        {
            json unsupported{{"ok", false},
                             {"error", "unsupported-shm-protocol"},
                             {"id", id},
                             {"shmProtocolVersion", kSharedMemoryProtocolVersion},
                             {"requestedProtocol", shmProtocol}};
            auto responseBytes = json::to_msgpack(unsupported);
            sendPacket(client, responseBytes);
            continue;
        }

        json reply = safeCall([&]() -> json
                              {
            if (cmd == "respond") {
                auto rt = getRuntime(msg.value("controllerName", ""));
                return rt->processInput(msg.value("payload", json::object()));
            }
            if (cmd == "metrics") {
                auto rt = getRuntime(msg.value("controllerName", ""));
                auto name = msg.value("controllerName", "");
                return json{{"name", name}, {"online", true}, {"health", json{{"status", "ok"}, {"since", nowEpochMs()}, {"failures", 0}}}, {"metrics", rt->metrics()}, {"params", rt->cloneParams()}};
            }
            if (cmd == "cloneParams") {
                auto rt = getRuntime(msg.value("controllerName", ""));
                return rt->cloneParams();
            }
            if (cmd == "applyParams") {
                auto rt = getRuntime(msg.value("controllerName", ""));
                rt->setParams(msg.value("params", json::object()));
                return true;
            }
            if (cmd == "snapshot") {
                auto rt = getRuntime(msg.value("controllerName", ""));
                return rt->toSnapshot();
            }
            if (cmd == "applySnapshot") {
                auto rt = getRuntime(msg.value("controllerName", ""));
                rt->fromSnapshot(msg.value("snapshot", json::object()));
                return true;
            }
            if (cmd == "applySnapshotAll") {
                auto snap = msg.value("snapshot", json::object());
                for (auto &name : controllerNames) {
                    getRuntime(name)->fromSnapshot(snap);
                }
                return true;
            }
            if (cmd == "listMetrics") {
                json out = json::array();
                for (auto &name : controllerNames) {
                    auto rt = getRuntime(name);
                    out.push_back(json{{"name", name}, {"online", true}, {"health", json{{"status", "ok"}, {"since", nowEpochMs()}, {"failures", 0}}}, {"metrics", rt->metrics()}, {"params", rt->cloneParams()}});
                }
                return out;
            }
            if (cmd == "ingestDocument") {
                json results = json::array();
                auto doc = msg.value("doc", json::object());
                for (auto &name : controllerNames) {
                    results.push_back(getRuntime(name)->ingestDocument(doc));
                }
                return results;
            }
            if (cmd == "ingestDocumentTo") {
                auto rt = getRuntime(msg.value("controllerName", ""));
                return rt->ingestDocument(msg.value("doc", json::object()));
            }
            if (cmd == "ingestDocumentToGroup") {
                json results = json::array();
                auto doc = msg.value("doc", json::object());
                for (auto &name : controllerNames) {
                    results.push_back(getRuntime(name)->ingestDocument(doc));
                }
                return results;
            }
            if (cmd == "forgetMemes") {
                json results = json::array();
                auto criteria = msg.value("criteria", json::object());
                for (auto &name : controllerNames) {
                    results.push_back(getRuntime(name)->forgetMemes(criteria));
                }
                return results;
            }
            if (cmd == "runtimeCall") {
                auto rt = getRuntime(msg.value("controllerName", ""));
                std::string method = msg.value("method", "");
                auto args = msg.value("args", json::array());
                if (method == "onlineLookup") {
                    auto a0 = (args.is_array() && args.size() > 0) ? args[0] : json::object();
                    auto a1 = (args.is_array() && args.size() > 1) ? args[1] : json::object();
                    return rt->onlineLookup(a0, a1);
                }
                if (method == "collectRobotsDocuments") {
                    json out = json::array();
                    auto a0 = (args.is_array() && args.size() > 0) ? args[0] : json::object();
                    for (auto &d : rt->collectRobotsDocuments(a0)) {
                        out.push_back(json{{"id", d.id}, {"text", d.text}, {"tokens", d.tokens}, {"source", d.source}, {"file", d.file}});
                    }
                    return out;
                }
                if (method == "listRobotsFiles") return rt->listRobotsFiles();
                if (method == "exportGraphToFile") {
                    auto a0 = (args.is_array() && args.size() > 0) ? args[0] : json::object();
                    auto p = rt->exportGraphToFile(a0);
                    return json{{"path", p.string()}};
                }
                if (method == "getSearchConfig") return rt->getSearchConfig();
                if (method == "setSearchConfig") {
                    auto a0 = (args.is_array() && args.size() > 0) ? args[0] : json::object();
                    return rt->setSearchConfig(a0);
                }
                if (method == "learnFromDialog") {
                    if (args.is_array() && args.size() == 1 && args[0].is_object()) {
                        auto obj = args[0];
                        return rt->learnFromDialog(obj.value("payload", json::object()), obj.value("result", json::object()));
                    }
                    auto a0 = (args.is_array() && args.size() > 0) ? args[0] : json::object();
                    auto a1 = (args.is_array() && args.size() > 1) ? args[1] : json::object();
                    return rt->learnFromDialog(a0, a1);
                }
                throw std::runtime_error("runtimeCall not allowed: " + method);
            }
            if (cmd == "memebarrierStart") {
                double th = msg.value("maliciousThreshold", MODEL_DEFAULTS.maliciousThreshold);
                if (!memeBarrier) {
                    memeBarrier = std::make_shared<MemeBarrier>(getRuntime(controllerNames.front()), 10000, th, 5);
                }
                memeBarrier->setThreshold(th);
                memeBarrier->start();
                auto stats = memeBarrier->getStats();
                return json{{"running", true}, {"threshold", memeBarrier->threshold()}, {"stats", json{{"scans", stats.scans}, {"isolated", stats.isolated}, {"lastScanTime", stats.lastScanTime}, {"lastIsolated", stats.lastIsolated}, {"evaluated", stats.evaluated}, {"candidates", stats.candidates}, {"avgScore", stats.avgScore}, {"maxScore", stats.maxScore}, {"p90Score", stats.p90Score}, {"lastThreshold", stats.lastThreshold}, {"avgDegree", stats.avgDegree}, {"stdDegree", stats.stdDegree}, {"avgOutDegree", stats.avgOutDegree}, {"stdOutDegree", stats.stdOutDegree}, {"scoreHistogram", stats.scoreHistogram}}}};
            }
            if (cmd == "memebarrierStop") {
                if (memeBarrier) memeBarrier->stop();
                return json{{"running", false}};
            }
            if (cmd == "memebarrierStats") {
                if (!memeBarrier) return json{{"running", false}, {"threshold", MODEL_DEFAULTS.maliciousThreshold}, {"stats", nullptr}};
                auto stats = memeBarrier->getStats();
                return json{{"running", memeBarrier->running()}, {"threshold", memeBarrier->threshold()}, {"stats", json{{"scans", stats.scans}, {"isolated", stats.isolated}, {"lastScanTime", stats.lastScanTime}, {"lastIsolated", stats.lastIsolated}, {"evaluated", stats.evaluated}, {"candidates", stats.candidates}, {"avgScore", stats.avgScore}, {"maxScore", stats.maxScore}, {"p90Score", stats.p90Score}, {"lastThreshold", stats.lastThreshold}, {"avgDegree", stats.avgDegree}, {"stdDegree", stats.stdDegree}, {"avgOutDegree", stats.avgOutDegree}, {"stdOutDegree", stats.stdOutDegree}, {"scoreHistogram", stats.scoreHistogram}}}};
            }
            if (cmd == "memebarrierSetThreshold") {
                double th = msg.value("maliciousThreshold", MODEL_DEFAULTS.maliciousThreshold);
                if (!memeBarrier) {
                    memeBarrier = std::make_shared<MemeBarrier>(getRuntime(controllerNames.front()), 10000, th, 5);
                }
                memeBarrier->setThreshold(th);
                return json{{"threshold", memeBarrier->threshold()}};
            }
            throw std::runtime_error("unknown-cmd:" + cmd); });

        reply["id"] = id;
        reply["shmProtocolVersion"] = kSharedMemoryProtocolVersion;
        reply["transport"] = "socket-msgpack-shm2";
        reply["groupId"] = groupId;
        auto responseBytes = json::to_msgpack(reply);
        if (!sendPacket(client, responseBytes))
            break;
    }

    closeSocket(client);
    closeSocket(serverSock);
    return 0;
}

static bool spawnFrontendProcess()
{
    int spawnDepth = std::max(0, (int)numberOr(getEnv("AI_FRONTEND_SPAWN_DEPTH", "0"), 0));
    if (spawnDepth >= 2)
        return false;
    std::string nextDepth = std::to_string(spawnDepth + 1);
    std::string projectRoot = getProjectRoot().string();
#ifdef _WIN32
    char oldSubBuf[64] = {0};
    char oldDepthBuf[64] = {0};
    char oldFrontendHttpLogBuf[64] = {0};
    DWORD oldSubLen = GetEnvironmentVariableA("AI_FRONTEND_SUBPROCESS", oldSubBuf, (DWORD)sizeof(oldSubBuf));
    DWORD oldDepthLen = GetEnvironmentVariableA("AI_FRONTEND_SPAWN_DEPTH", oldDepthBuf, (DWORD)sizeof(oldDepthBuf));
    DWORD oldFrontendHttpLogLen = GetEnvironmentVariableA("FRONTEND_HTTP_LOG", oldFrontendHttpLogBuf, (DWORD)sizeof(oldFrontendHttpLogBuf));
    bool hadSub = oldSubLen > 0 && oldSubLen < sizeof(oldSubBuf);
    bool hadDepth = oldDepthLen > 0 && oldDepthLen < sizeof(oldDepthBuf);
    bool hadFrontendHttpLog = oldFrontendHttpLogLen > 0 && oldFrontendHttpLogLen < sizeof(oldFrontendHttpLogBuf);

    SetEnvironmentVariableA("AI_FRONTEND_SUBPROCESS", "1");
    SetEnvironmentVariableA("AI_FRONTEND_SPAWN_DEPTH", nextDepth.c_str());
    STARTUPINFOA si{};
    PROCESS_INFORMATION pi{};
    si.cb = sizeof(si);
    std::string cmdLine = std::string("\"") + g_selfPath + "\" --frontend-only=1";
    BOOL ok = CreateProcessA(nullptr, cmdLine.data(), nullptr, nullptr, FALSE, CREATE_NEW_PROCESS_GROUP, nullptr,
                             projectRoot.empty() ? nullptr : projectRoot.c_str(), &si, &pi);

    if (hadSub)
        SetEnvironmentVariableA("AI_FRONTEND_SUBPROCESS", oldSubBuf);
    else
        SetEnvironmentVariableA("AI_FRONTEND_SUBPROCESS", nullptr);
    if (hadDepth)
        SetEnvironmentVariableA("AI_FRONTEND_SPAWN_DEPTH", oldDepthBuf);
    else
        SetEnvironmentVariableA("AI_FRONTEND_SPAWN_DEPTH", nullptr);
    if (hadFrontendHttpLog)
        SetEnvironmentVariableA("FRONTEND_HTTP_LOG", oldFrontendHttpLogBuf);
    else
        SetEnvironmentVariableA("FRONTEND_HTTP_LOG", nullptr);

    if (!ok)
        return false;
    {
        std::lock_guard<std::mutex> lock(gFrontendProcMu);
        if (gFrontendProcessHandle)
        {
            DWORD waitCode = WaitForSingleObject(gFrontendProcessHandle, 0);
            if (waitCode == WAIT_TIMEOUT)
            {
                TerminateProcess(gFrontendProcessHandle, 0);
                WaitForSingleObject(gFrontendProcessHandle, 1500);
            }
            CloseHandle(gFrontendProcessHandle);
            gFrontendProcessHandle = nullptr;
            gFrontendProcessId = 0;
        }
        gFrontendProcessHandle = pi.hProcess;
        gFrontendProcessId = pi.dwProcessId;
        ensureFrontendJobObjectLocked();
        if (gFrontendJobHandle)
            AssignProcessToJobObject(gFrontendJobHandle, gFrontendProcessHandle);
    }
    CloseHandle(pi.hThread);
    return true;
#else
    setenv("AI_FRONTEND_SUBPROCESS", "1", 1);
    setenv("AI_FRONTEND_SPAWN_DEPTH", nextDepth.c_str(), 1);
    pid_t pid = fork();
    if (pid == 0)
    {
        if (!projectRoot.empty())
        {
            std::error_code ec;
            fs::current_path(projectRoot, ec);
        }
        execl(g_selfPath.c_str(), g_selfPath.c_str(), "--frontend-only=1", nullptr);
        _exit(1);
    }
    if (pid > 0)
    {
        std::lock_guard<std::mutex> lock(gFrontendProcMu);
        gFrontendProcessPid = pid;
        return true;
    }
    return false;
#endif
}

// main 为程序主入口，负责初始化与模式分派。
// 调用方式：进程启动后由操作系统直接调用。
// 实现思路：加载配置、构建存储与池化组件后启动网关。
// 注意事项：命令行参数可改变进程运行模式。
int main(int argc, char **argv)
{   
    std::cout << "AI Controller starting..." << std::endl;
    g_selfPath = fs::absolute(fs::path(argv[0])).string();
    std::atexit(cleanupFrontendProcess);
    std::atexit(cleanupPidFile);
#ifdef _WIN32
    SetConsoleCtrlHandler(onConsoleControl, TRUE);
    SetUnhandledExceptionFilter(onUnhandledException);
#endif
    installFatalHooks(fs::path("runtime_store"));
    Config config;
    std::map<std::string, std::string> args;
    try
    {
        args = parseArgs(argc, argv);

        auto parseIntArgOrEnv = [&](const std::string &argKey, const std::string &envKey, int fallback)
        {
            auto it = args.find(argKey);
            std::string raw = (it != args.end() && !it->second.empty()) ? it->second : getEnv(envKey, std::to_string(fallback));
            try
            {
                return std::stoi(raw);
            }
            catch (...)
            {
                return fallback;
            }
        };
        int forceExitMs = parseIntArgOrEnv("sigint-force-exit-ms", "AI_SIGINT_FORCE_EXIT_MS", 12000);
        forceExitMs = std::max(1000, std::min(forceExitMs, 600000));
#ifdef _WIN32
        gCtrlForceExitMs.store(static_cast<DWORD>(forceExitMs), std::memory_order_relaxed);
#endif

        applyFrontendEnvFromArgs(args);

        bool frontendOnlyRequested = args.count("frontend-only") != 0 || boolFrom(getEnv("AI_FRONTEND_SUBPROCESS", "false"), false);
        if (frontendOnlyRequested)
        {
            try
            {
                setupFrontendServer();
                drogon::app().run();
                return 0;
            }
            catch (const std::exception &e)
            {
                std::cerr << "frontend-only exception: " << e.what() << std::endl;
                return 1;
            }
            catch (...)
            {
                std::cerr << "frontend-only unknown exception" << std::endl;
                return 1;
            }
        }
        fs::path pidPath = fs::path(getEnv("AI_PID_FILE", (fs::path("runtime_store") / "phoenix_main.pid").string()));
        if (args.count("pid-file"))
            pidPath = fs::path(args["pid-file"]);
        std::string pidError;
        if (!acquirePidFile(pidPath, pidError))
        {
            std::cerr << "startup blocked: " << pidError << std::endl;
            return 1;
        }
        std::cout << "pid file locked: " << pidPath.string() << std::endl;
        initProcessMemoryReservation();
        config = loadConfig(argc, argv);
        applyFrontendEnvFromConfig(config);
        installFatalHooks(config.baseDir);
    }
    catch (const std::exception &e)
    {
        std::cerr << "startup exception: " << e.what() << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << "startup unknown exception" << std::endl;
        return 1;
    }
    std::cout << "Configuration loaded. Starting in " << (args.count("group-worker") ? "group worker" : "gateway") << " mode." << std::endl;
    std::cout << "startup-flags usingOllama=" << (config.usingOllama ? "true" : "false")
              << " httpLog=" << (config.httpLog ? "true" : "false")
              << " port=" << config.portGateway
              << " studyPort=" << config.portStudy << std::endl;
#ifdef _WIN32
    if (config.httpLog)
        SetEnvironmentVariableA("AI_HTTP_LOG", "true");
#else
    if (config.httpLog)
        setenv("AI_HTTP_LOG", "true", 1);
#endif
    try
    {
        auto cwd = fs::current_path();
        auto root = getProjectRoot();
        std::string diag = std::string("startup-paths cwd=") + cwd.string() +
                           " projectRoot=" + root.string() +
                           " robotsDir=" + config.robotsDir.string();
        std::cout << diag << std::endl;
        appendFatalLog(diag);
    }
    catch (...)
    {
    }
    if (args.count("group-worker"))
    {
        return runGroupWorker(config);
    }
    ensureDir(config.baseDir);
    ensureDir(config.snapshotDir);
    ensureDir(config.lmdbRoot);
    ensureDir(config.robotsDir);
    if (!config.dbPath.empty())
    {
        ensureDir(config.dbPath.parent_path());
    }
    std::cout << "Directories ensured. Initializing storage..." << std::endl;
    std::shared_ptr<KeyValueStore> kvmStore;
    std::shared_ptr<KeyValueStore> memeStore;
    std::shared_ptr<KeyValueStore> sessionStore;

    std::shared_ptr<Database079> db079;
#ifdef HAVE_SQLITE
    db079 = std::make_shared<Database079>(config.dbPath, config.lmdbRoot, config.redisUrl, config.redisCacheDb, config.redisCachePrefix);
    if (db079->open())
    {
        kvmStore = db079->createStore("kvm");
        memeStore = db079->createStore("meme_graph");
        sessionStore = db079->createStore("session");
    }
#endif
    std::cout << "Storage initialized. Starting gateway server..." << std::endl;
#ifdef HAVE_LMDB
    if (!Database079::isSqliteStore(kvmStore))
    {
        auto lmdbK = std::make_shared<LmdbStore>("kvm", config.lmdbRoot, config.lmdbMapSizeBytes);
        auto lmdbM = std::make_shared<LmdbStore>("meme_graph", config.lmdbRoot, config.lmdbMapSizeBytes);
        auto lmdbS = std::make_shared<LmdbStore>("session", config.lmdbRoot, config.lmdbMapSizeBytes);
        if (lmdbK->ok())
            kvmStore = lmdbK;
        if (lmdbM->ok())
            memeStore = lmdbM;
        if (lmdbS->ok())
            sessionStore = lmdbS;
    }
#endif
    std::cout << "Gateway server storage setup complete. Starting application..." << std::endl;
    if (!kvmStore)
    {
        kvmStore = std::make_shared<JsonFileStore>("kvm", config.lmdbRoot);
        memeStore = std::make_shared<JsonFileStore>("meme_graph", config.lmdbRoot);
        sessionStore = std::make_shared<JsonFileStore>("session", config.lmdbRoot);
    }

    std::shared_ptr<ControllerPoolBase> pool;
    std::shared_ptr<IRedisSynchronizer> redisSync;
    std::shared_ptr<RotationManager> rotation;
    std::shared_ptr<IStudyEngine> study;
    std::cout << "Storage setup complete. Initializing controller pool..." << std::endl;
    if (config.groupProc)
    {
        pool = std::make_shared<GroupProcessPool>(kvmStore, memeStore, sessionStore, config);
        auto redisFactory = module_mount::ModuleRegistry::instance().redisSynchronizerFactory();
        if (redisFactory)
            redisSync = redisFactory(pool, config.redisUrl, config.redisChannel);
        else
            redisSync = std::make_shared<RedisSynchronizer>(pool, config.redisUrl, config.redisChannel);
        std::cout << "[Bootstrap] group-proc enabled: one process per group" << std::endl;
    }
    else
    {
        pool = std::make_shared<ControllerPool>(kvmStore, memeStore, sessionStore, config);
        rotation = std::make_shared<RotationManager>(pool);
        rotation->start();
        auto redisFactory = module_mount::ModuleRegistry::instance().redisSynchronizerFactory();
        if (redisFactory)
            redisSync = redisFactory(pool, config.redisUrl, config.redisChannel);
        else
            redisSync = std::make_shared<RedisSynchronizer>(pool, config.redisUrl, config.redisChannel);
        try
        {
            redisSync->start();
        }
        catch (...)
        {
        }
        auto studyFactory = module_mount::ModuleRegistry::instance().studyEngineFactory();
        if (studyFactory)
            study = studyFactory(pool, redisSync);
        else
            study = std::make_shared<StudyEngine>(pool, redisSync);
        study->start();
    }

    std::shared_ptr<ISnapshotManager> snapshots;
    auto snapshotFactory = module_mount::ModuleRegistry::instance().snapshotManagerFactory();
    if (snapshotFactory)
        snapshots = snapshotFactory(pool->getActive()->runtime(), config.snapshotDir);
    else
        snapshots = std::make_shared<SnapshotManager>(pool->getActive()->runtime(), config.snapshotDir);

    bool restoredFromSnapshot = false;
    std::cout << "Controller pool initialized. Attempting to restore from snapshot..." << std::endl;
    try
    {
        auto list = snapshots->list();
        std::sort(list.begin(), list.end(), std::greater<>());
        if (!list.empty())
        {
            auto snap = snapshots->restore(list[0]);
            restoredFromSnapshot = true;
            if (config.groupProc)
            {
                if (auto gp = dynamic_cast<GroupProcessPool *>(pool.get()))
                {
                    try
                    {
                        gp->applySnapshotAll(snap);
                    }
                    catch (...)
                    {
                    }
                }
            }
            if (config.syncStandbyOnBoot)
            {
                try
                {
                    pool->getStandby()->applySnapshot(snap);
                }
                catch (...)
                {
                }
                try
                {
                    pool->getValidation()->applySnapshot(snap);
                }
                catch (...)
                {
                }
            }
        }
    }
    catch (...)
    {
    }

    if (config.robotsAutoload && config.robotsWarmupLimit > 0 && !restoredFromSnapshot)
    {
        std::cout << "Robots autoload enabled. Warming up robots..." << std::endl;
        try
        {
            auto docs = pool->getActive()->runtime()->collectRobotsDocuments(json{{"limit", config.robotsWarmupLimit}, {"shuffle", config.robotsWarmupShuffle}});
            auto groups = pool->listGroupIds();
            for (auto &doc : docs)
            {
                std::string key = (doc.source.empty() ? (doc.file.empty() ? doc.id : doc.file) : doc.source);
                key += "|" + doc.text.substr(0, 256);
                int idx = groups.empty() ? 0 : (int)(hashStrSimple(key) % groups.size());
                std::string target = groups.empty() ? "G1" : (groups[idx].empty() ? (groups.empty() ? "G1" : groups[0]) : groups[idx]);
                std::string source = doc.source.empty() ? (doc.file.empty() ? "robots:unknown" : ("robots:" + doc.file)) : doc.source;
                pool->ingestDocumentToGroup(target, json{{"text", doc.text}, {"tokens", doc.tokens}, {"source", source}, {"file", doc.file}, {"id", doc.id}});
            }
        }
        catch (...)
        {
        }
    }
    std::cout << "Snapshot restoration complete. Starting gateway server..." << std::endl;
    if (config.testsAutoload)
    {
        std::thread([pool, testsDir = config.testsDir]()
                    {
            try
            {
                if (!fs::exists(testsDir))
                    return;

                std::vector<std::string> files;
                for (auto &e : fs::directory_iterator(testsDir))
                {
                    if (!e.is_regular_file())
                        continue;
                    auto ext = e.path().extension().string();
                    std::string lowerExt;
                    lowerExt.reserve(ext.size());
                    for (char c : ext)
                        lowerExt.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
                    if (lowerExt == ".txt")
                        files.push_back(e.path().filename().string());
                }

                std::sort(files.begin(), files.end());
                const size_t maxFiles = 200;
                const size_t maxFileBytes = 2ull * 1024ull * 1024ull;
                if (files.size() > maxFiles)
                    files.resize(maxFiles);

                auto groups = pool->listGroupIds();
                for (auto &f : files)
                {
                    fs::path p = testsDir / f;
                    std::error_code ec;
                    auto sz = fs::file_size(p, ec);
                    if (!ec && sz > maxFileBytes)
                        continue;

                    std::string text = readTextFile(p);
                    if (text.empty())
                        continue;
                    std::string key = "tests:" + f;

                    if (groups.empty())
                    {
                        pool->ingestDocument(json{{"text", text}, {"source", key}});
                        continue;
                    }

                    int idx = (int)(hashStrSimple(key) % groups.size());
                    std::string target = groups[idx].empty() ? groups[0] : groups[idx];
                    pool->ingestDocumentToGroup(target, json{{"text", text}, {"source", key}});
                }
            }
            catch (...)
            {
            } })
            .detach();
    }
    std::cout << "Gateway server starting..." << std::endl;
    std::shared_ptr<IGatewayServer> gateway;
    auto gatewayFactory = module_mount::ModuleRegistry::instance().gatewayServerFactory();
    if (gatewayFactory)
        gateway = gatewayFactory(pool, snapshots, redisSync, rotation, study, db079, config);
    else
        gateway = std::make_shared<GatewayServer>(pool, snapshots, redisSync, rotation, study, db079, config);
    {
        bool frontendEnabled = boolFrom(getEnv("AI_FRONTEND_ENABLED", "true"), true);
        bool isFrontendSubprocess = boolFrom(getEnv("AI_FRONTEND_SUBPROCESS", "false"), false);
        if (frontendEnabled && !isFrontendSubprocess)
        {
            if (!spawnFrontendProcess())
            {
                std::cerr << "frontend subprocess spawn failed" << std::endl;
            }
        }
    }
    if (config.learningWarmup)
    {
        std::thread([gateway]()
                    {
            try { gateway->warmupLearning(); } catch (...) {} })
            .detach();
    }
    if (config.transformerBootstrap)
    {
        int bootstrapDocs = std::max(16, config.transformerBootstrapDocs);
        std::thread([gateway, bootstrapDocs]()
                    {
            try { gateway->bootstrapTransformerFromCorpus(bootstrapDocs); } catch (...) {} })
            .detach();
    }
    std::cout << "Initialization complete. Listening for requests..." << std::endl;
    try
    {
        gateway->listen();
        return 0;
    }
    catch (const std::exception &e)
    {
        appendFatalLog(std::string("gateway listen exception: ") + e.what());
        std::cerr << "gateway listen exception: " << e.what() << std::endl;
        return 1;
    }
    catch (...)
    {
        appendFatalLog("gateway listen unknown exception");
        std::cerr << "gateway listen unknown exception" << std::endl;
        return 1;
    }
}
