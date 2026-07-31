/* external_runtime.cpp - External runtime implementation
   Copyright (C) 2026 079 Project

   This file is part of 079 Project.

   079 Project is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   079 Project is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with 079 Project.  If not, see <http://www.gnu.org/licenses/>. */

#include "external_runtime.hpp"
#include "gguf_tensor_parser.hpp"

#include <drogon/HttpClient.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <future>
#include <iomanip>
#include <sstream>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <winsock2.h>
#include <windows.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <cerrno>
#include <fcntl.h>
#include <netdb.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace external_runtime {

namespace {

int64_t nowMs() {
    return static_cast<int64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
}

std::string trimCopy(const std::string &input) {
    const auto begin = input.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return {};
    }
    const auto end = input.find_last_not_of(" \t\r\n");
    return input.substr(begin, end - begin + 1);
}

bool writeJsonFile(const fs::path &path, const json &doc) {
    std::error_code ec;
    fs::create_directories(path.parent_path(), ec);
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        return false;
    }
    out << doc.dump(2);
    return static_cast<bool>(out);
}

bool isFreshStatusDocument(const json &doc, int maxAgeMs, int64_t now, int64_t &updatedAtMs, int64_t &ageMs) {
    updatedAtMs = doc.is_object() ? doc.value("updatedAtMs", static_cast<int64_t>(0)) : 0;
    if (updatedAtMs <= 0) {
        ageMs = -1;
        return false;
    }
    ageMs = std::max<int64_t>(0, now - updatedAtMs);
    return ageMs <= std::max(1000, maxAgeMs);
}

std::string sanitizeName(std::string value) {
    for (char &ch : value) {
        if (!(std::isalnum(static_cast<unsigned char>(ch)) || ch == '-' || ch == '_')) {
            ch = '_';
        }
    }
    return value;
}

std::string formatDouble(double value) {
    std::ostringstream out;
    out << std::setprecision(12) << value;
    return out.str();
}

uint64_t roundDownPowerOfTwo(uint64_t value) {
    if (value == 0) {
        return 0;
    }
    uint64_t rounded = 1;
    while (rounded <= value / 2) {
        rounded <<= 1;
    }
    return rounded;
}

uint64_t freePhysicalMemoryBytes() {
#ifdef _WIN32
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (!GlobalMemoryStatusEx(&status)) {
        return 0;
    }
    return static_cast<uint64_t>(status.ullAvailPhys);
#else
    return 0;
#endif
}

int64_t findJsonScalarBySuffix(const json &object, const std::vector<std::string> &suffixes) {
    if (!object.is_object()) {
        return 0;
    }
    for (auto it = object.begin(); it != object.end(); ++it) {
        std::string key = it.key();
        std::transform(key.begin(), key.end(), key.begin(), [](unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
        });
        for (const auto &suffix : suffixes) {
            if (key.size() < suffix.size() || key.compare(key.size() - suffix.size(), suffix.size(), suffix) != 0) {
                continue;
            }
            if (it.value().is_number_integer() || it.value().is_number_unsigned()) {
                return it.value().get<int64_t>();
            }
            if (it.value().is_number_float()) {
                return static_cast<int64_t>(it.value().get<double>());
            }
        }
    }
    return 0;
}

int64_t inferKvBytesPerToken(const gguf_tensor_parser::InspectResult &inspection) {
    if (!inspection.valid || !inspection.report.is_object()) {
        return 0;
    }
    const json model = inspection.report.value("model", json::object());
    const json interesting = inspection.report.value("kv", json::object()).value("interesting", json::object());
    const int64_t blockCount = std::max<int64_t>(0, model.value("blockCount", static_cast<int64_t>(0)));
    const int64_t embeddingWidth = std::max<int64_t>(0, model.value("embeddingWidth", static_cast<int64_t>(0)));
    const int64_t attentionHeads = std::max<int64_t>(0, model.value("attentionHeads", static_cast<int64_t>(0)));
    int64_t attentionHeadsKv = std::max<int64_t>(0, findJsonScalarBySuffix(interesting, {".attention.head_count_kv", ".n_head_kv"}));
    if (attentionHeadsKv <= 0) {
        attentionHeadsKv = attentionHeads;
    }
    if (blockCount <= 0 || embeddingWidth <= 0 || attentionHeads <= 0 || attentionHeadsKv <= 0) {
        return 0;
    }
    const int64_t kvWidthPerLayer = std::max<int64_t>(1, (embeddingWidth * attentionHeadsKv) / attentionHeads);
    return 4LL * blockCount * kvWidthPerLayer;
}

void appendIntArg(std::ostringstream &cmd, const std::string &name, int value) {
    if (value > 0) {
        cmd << ' ' << name << ' ' << value;
    }
}

void appendDoubleArg(std::ostringstream &cmd, const std::string &name, double value) {
    cmd << ' ' << name << ' ' << formatDouble(value);
}

bool parseHttpUrl(const std::string &url, std::string &host, int &port) {
    std::string working = trimCopy(url);
    const std::string http = "http://";
    const std::string https = "https://";
    if (working.rfind(http, 0) == 0) {
        working = working.substr(http.size());
        port = 80;
    } else if (working.rfind(https, 0) == 0) {
        working = working.substr(https.size());
        port = 443;
    } else {
        port = 80;
    }
    const auto slashPos = working.find('/');
    if (slashPos != std::string::npos) {
        working = working.substr(0, slashPos);
    }
    const auto colonPos = working.rfind(':');
    if (colonPos != std::string::npos && colonPos + 1 < working.size()) {
        host = working.substr(0, colonPos);
        try {
            port = std::stoi(working.substr(colonPos + 1));
        } catch (...) {
            return false;
        }
    } else {
        host = working;
    }
    host = trimCopy(host);
    return !host.empty() && port > 0;
}

#ifdef _WIN32
class WsaSession {
public:
    WsaSession() {
        WSADATA data{};
        ok_ = (WSAStartup(MAKEWORD(2, 2), &data) == 0);
    }
    ~WsaSession() {
        if (ok_) {
            WSACleanup();
        }
    }
    bool ok() const { return ok_; }
private:
    bool ok_{false};
};

bool tcpReady(const std::string &host, int port, int timeoutMs) {
    WsaSession session;
    if (!session.ok()) {
        return false;
    }
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;
    addrinfo *result = nullptr;
    const std::string portStr = std::to_string(port);
    if (getaddrinfo(host.c_str(), portStr.c_str(), &hints, &result) != 0) {
        return false;
    }
    bool ready = false;
    for (addrinfo *ptr = result; ptr != nullptr && !ready; ptr = ptr->ai_next) {
        SOCKET sock = socket(ptr->ai_family, ptr->ai_socktype, ptr->ai_protocol);
        if (sock == INVALID_SOCKET) {
            continue;
        }
        u_long nonBlocking = 1;
        ioctlsocket(sock, FIONBIO, &nonBlocking);
        int rc = connect(sock, ptr->ai_addr, static_cast<int>(ptr->ai_addrlen));
        if (rc == 0) {
            ready = true;
        } else {
            fd_set wfds;
            FD_ZERO(&wfds);
            FD_SET(sock, &wfds);
            timeval tv{};
            tv.tv_sec = timeoutMs / 1000;
            tv.tv_usec = (timeoutMs % 1000) * 1000;
            rc = select(0, nullptr, &wfds, nullptr, &tv);
            if (rc > 0 && FD_ISSET(sock, &wfds)) {
                int error = 0;
                int len = sizeof(error);
                getsockopt(sock, SOL_SOCKET, SO_ERROR, reinterpret_cast<char *>(&error), &len);
                ready = (error == 0);
            }
        }
        closesocket(sock);
    }
    freeaddrinfo(result);
    return ready;
}

uint32_t launchDetachedProcess(const std::string &commandLine, const fs::path &workingDir) {
    STARTUPINFOA si{};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};
    std::string mutableCmd = commandLine;
    BOOL ok = CreateProcessA(
        nullptr,
        mutableCmd.data(),
        nullptr,
        nullptr,
        FALSE,
        CREATE_NO_WINDOW | DETACHED_PROCESS,
        nullptr,
        workingDir.empty() ? nullptr : workingDir.string().c_str(),
        &si,
        &pi);
    if (!ok) {
        return 0;
    }
    const uint32_t pid = pi.dwProcessId;
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    return pid;
}

bool httpEndpointReady(const std::string &baseUrl, const std::string &path, int timeoutMs) {
    auto client = drogon::HttpClient::newHttpClient(baseUrl);
    if (!client) {
        return false;
    }
    auto req = drogon::HttpRequest::newHttpRequest();
    req->setMethod(drogon::Get);
    req->setPath(path);

    std::promise<std::pair<drogon::ReqResult, drogon::HttpResponsePtr>> promise;
    auto future = promise.get_future();
    client->sendRequest(req, [&promise](drogon::ReqResult result, const drogon::HttpResponsePtr &resp) {
        try {
            promise.set_value({result, resp});
        } catch (...) {
        }
    });

    if (future.wait_for(std::chrono::milliseconds(std::max(100, timeoutMs))) != std::future_status::ready) {
        return false;
    }
    const auto pair = future.get();
    return pair.first == drogon::ReqResult::Ok && pair.second &&
           pair.second->statusCode() >= 200 && pair.second->statusCode() < 300;
}

bool backendReady(const BackendRuntimeSpec &spec, const std::string &host, int port, int timeoutMs) {
    if (!tcpReady(host, port, timeoutMs)) {
        return false;
    }
    if (spec.provider == "llamacpp") {
        if (!httpEndpointReady(spec.baseUrl, "/health", timeoutMs)) {
            return false;
        }
        if (httpEndpointReady(spec.baseUrl, "/props", timeoutMs)) {
            return true;
        }
        return httpEndpointReady(spec.baseUrl, "/v1/models", timeoutMs);
    }
    return true;
}
#else
bool tcpReady(const std::string &host, int port, int timeoutMs) {
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;
    addrinfo *result = nullptr;
    const std::string portText = std::to_string(port);
    if (getaddrinfo(host.c_str(), portText.c_str(), &hints, &result) != 0) {
        return false;
    }
    bool ready = false;
    for (addrinfo *entry = result; entry != nullptr && !ready; entry = entry->ai_next) {
        const int socketFd = socket(entry->ai_family, entry->ai_socktype, entry->ai_protocol);
        if (socketFd < 0) {
            continue;
        }
        const int flags = fcntl(socketFd, F_GETFL, 0);
        if (flags >= 0) {
            fcntl(socketFd, F_SETFL, flags | O_NONBLOCK);
        }
        const int connectResult = connect(socketFd, entry->ai_addr, entry->ai_addrlen);
        if (connectResult == 0) {
            ready = true;
        } else if (errno == EINPROGRESS) {
            fd_set writable;
            FD_ZERO(&writable);
            FD_SET(socketFd, &writable);
            timeval wait{};
            wait.tv_sec = std::max(0, timeoutMs) / 1000;
            wait.tv_usec = (std::max(0, timeoutMs) % 1000) * 1000;
            if (select(socketFd + 1, nullptr, &writable, nullptr, &wait) > 0 && FD_ISSET(socketFd, &writable)) {
                int socketError = 0;
                socklen_t errorSize = sizeof(socketError);
                ready = getsockopt(socketFd, SOL_SOCKET, SO_ERROR, &socketError, &errorSize) == 0 && socketError == 0;
            }
        }
        close(socketFd);
    }
    freeaddrinfo(result);
    return ready;
}

bool httpEndpointReady(const std::string &baseUrl, const std::string &path, int timeoutMs) {
    auto client = drogon::HttpClient::newHttpClient(baseUrl);
    if (!client) {
        return false;
    }
    auto request = drogon::HttpRequest::newHttpRequest();
    request->setMethod(drogon::Get);
    request->setPath(path);
    std::promise<std::pair<drogon::ReqResult, drogon::HttpResponsePtr>> promise;
    auto future = promise.get_future();
    client->sendRequest(request, [&promise](drogon::ReqResult result, const drogon::HttpResponsePtr &response) {
        try {
            promise.set_value({result, response});
        } catch (...) {
        }
    });
    if (future.wait_for(std::chrono::milliseconds(std::max(100, timeoutMs))) != std::future_status::ready) {
        return false;
    }
    const auto response = future.get();
    return response.first == drogon::ReqResult::Ok && response.second &&
           response.second->statusCode() >= 200 && response.second->statusCode() < 300;
}

bool backendReady(const BackendRuntimeSpec &spec, const std::string &host, int port, int timeoutMs) {
    if (!tcpReady(host, port, timeoutMs)) {
        return false;
    }
    if (spec.provider != "llamacpp") {
        return true;
    }
    return httpEndpointReady(spec.baseUrl, "/health", timeoutMs) &&
           (httpEndpointReady(spec.baseUrl, "/props", timeoutMs) ||
            httpEndpointReady(spec.baseUrl, "/v1/models", timeoutMs));
}

uint32_t launchDetachedProcess(const std::string &, const fs::path &) {
    return 0;
}
#endif

fs::path resolveBinary(const BackendRuntimeSpec &spec) {
    std::vector<fs::path> candidates;
    const std::string provider = sanitizeName(spec.provider);
    candidates.push_back(spec.runtimeDir / (provider + "-server.exe"));
    candidates.push_back(spec.runtimeDir / provider / (provider + "-server.exe"));
    if (!spec.providerRoot.empty()) {
        candidates.push_back(spec.providerRoot / "build" / "bin" / (provider + "-server.exe"));
        candidates.push_back(spec.providerRoot / "build-gcc" / "bin" / (provider + "-server.exe"));
        candidates.push_back(spec.providerRoot / "bin" / (provider + "-server.exe"));
    }
    if (spec.provider == "llamacpp") {
        if (!spec.providerRoot.empty()) {
            candidates.push_back(spec.providerRoot / "build" / "bin" / "llama-server.exe");
            candidates.push_back(spec.providerRoot / "build-gcc" / "bin" / "llama-server.exe");
            candidates.push_back(spec.providerRoot / "build" / "bin" / "server.exe");
            candidates.push_back(spec.providerRoot / "build-gcc" / "bin" / "server.exe");
        }
        candidates.push_back(spec.runtimeDir / "llama-server.exe");
    } else if (spec.provider == "bitnet") {
        if (!spec.providerRoot.empty()) {
            candidates.push_back(spec.providerRoot / "build" / "bin" / "bitnet-server.exe");
            candidates.push_back(spec.providerRoot / "build-gcc" / "bin" / "bitnet-server.exe");
            candidates.push_back(spec.providerRoot / "build" / "bin" / "server.exe");
            candidates.push_back(spec.providerRoot / "build-gcc" / "bin" / "server.exe");
            candidates.push_back(spec.providerRoot / "main.exe");
        }
        candidates.push_back(spec.runtimeDir / "bitnet-server.exe");
    }
    for (const auto &candidate : candidates) {
        std::error_code ec;
        if (!candidate.empty() && fs::exists(candidate, ec) && fs::is_regular_file(candidate, ec)) {
            return fs::absolute(candidate);
        }
    }
    return {};
}

bool hasAdapterLauncher(const BackendRuntimeSpec &spec) {
    std::error_code ec;
    return !spec.pythonExecutable.empty() && !spec.adapterScriptPath.empty() &&
           fs::exists(spec.pythonExecutable, ec) && fs::is_regular_file(spec.pythonExecutable, ec) &&
           fs::exists(spec.adapterScriptPath, ec) && fs::is_regular_file(spec.adapterScriptPath, ec);
}

std::string substituteTemplate(std::string text,
                               const std::string &modelPath,
                               const std::string &host,
                               int port,
                               const fs::path &brainMap,
                               const BackendRuntimeSpec &spec) {
    std::string loraArgs;
    if (!spec.loraFiles.empty()) {
        std::istringstream iss(spec.loraFiles);
        std::string file;
        while (std::getline(iss, file, ',')) {
            file = trimCopy(file);
            if (!file.empty()) {
                loraArgs += " --lora \"" + file + "\"";
            }
        }
        if (spec.loraInitWithoutApply) {
            loraArgs += " --lora-init-without-apply";
        }
    }

    const std::vector<std::pair<std::string, std::string>> replacements = {
        {"{model}", modelPath},
        {"{host}", host},
        {"{port}", std::to_string(port)},
        {"{brain_map}", brainMap.string()},
        {"{provider}", spec.provider},
        {"{provider_root}", spec.providerRoot.string()},
        {"{runtime_dir}", spec.runtimeDir.string()},
        {"{python}", spec.pythonExecutable.string()},
        {"{adapter_script}", spec.adapterScriptPath.string()},
        {"{ctx_size}", std::to_string(spec.ctxSize)},
        {"{batch_size}", std::to_string(spec.batchSize)},
        {"{ubatch_size}", std::to_string(spec.ubatchSize)},
        {"{rope_scaling}", spec.ropeScaling},
        {"{rope_freq_base}", formatDouble(spec.ropeFreqBase)},
        {"{rope_freq_scale}", formatDouble(spec.ropeFreqScale)},
        {"{yarn_orig_ctx}", std::to_string(spec.yarnOrigCtx)},
        {"{yarn_ext_factor}", formatDouble(spec.yarnExtFactor)},
        {"{yarn_attn_factor}", formatDouble(spec.yarnAttnFactor)},
        {"{yarn_beta_fast}", formatDouble(spec.yarnBetaFast)},
        {"{yarn_beta_slow}", formatDouble(spec.yarnBetaSlow)},
        {"{lora_args}", loraArgs}};
    for (const auto &item : replacements) {
        std::string::size_type pos = 0;
        while ((pos = text.find(item.first, pos)) != std::string::npos) {
            text.replace(pos, item.first.size(), item.second);
            pos += item.second.size();
        }
    }
    return text;
}

std::string buildAdapterCommandLine(const BackendRuntimeSpec &spec,
                                    const std::string &host,
                                    int port,
                                    const fs::path &brainMap) {
    std::ostringstream cmd;
    cmd << '"' << spec.pythonExecutable.string() << '"'
        << " \"" << spec.adapterScriptPath.string() << "\""
        << " --provider \"" << spec.provider << "\""
        << " --engine-root \"" << spec.providerRoot.string() << "\""
        << " --runtime-dir \"" << spec.runtimeDir.string() << "\""
        << " --model \"" << spec.modelPath << "\""
        << " --host \"" << host << "\""
        << " --port " << port;
    if (!brainMap.empty()) {
        cmd << " --brain-map \"" << brainMap.string() << "\"";
    }
    appendIntArg(cmd, "--ctx-size", spec.ctxSize);
    appendIntArg(cmd, "--batch-size", spec.batchSize);
    appendIntArg(cmd, "--ubatch-size", spec.ubatchSize);
    if (!trimCopy(spec.ropeScaling).empty()) {
        cmd << " --rope-scaling " << spec.ropeScaling;
    }
    appendDoubleArg(cmd, "--rope-freq-base", spec.ropeFreqBase);
    appendDoubleArg(cmd, "--rope-freq-scale", spec.ropeFreqScale);
    appendIntArg(cmd, "--yarn-orig-ctx", spec.yarnOrigCtx);
    appendDoubleArg(cmd, "--yarn-ext-factor", spec.yarnExtFactor);
    appendDoubleArg(cmd, "--yarn-attn-factor", spec.yarnAttnFactor);
    appendDoubleArg(cmd, "--yarn-beta-fast", spec.yarnBetaFast);
    appendDoubleArg(cmd, "--yarn-beta-slow", spec.yarnBetaSlow);
    if (!trimCopy(spec.loraFiles).empty()) {
        cmd << " --lora-files \"" << spec.loraFiles << "\"";
    }
    if (spec.loraInitWithoutApply) {
        cmd << " --lora-init-without-apply";
    }
    return cmd.str();
}

json buildBrainMap(const BackendRuntimeSpec &spec,
                   const fs::path &outputPath,
                   gguf_tensor_parser::InspectResult *inspectionOut = nullptr) {
    const auto inspection = gguf_tensor_parser::inspectFile(spec.modelPath);
    if (inspectionOut) {
        *inspectionOut = inspection;
    }
    json doc = gguf_tensor_parser::buildBrainMapDocument(spec.provider,
                                                         spec.modelPath,
                                                         inspection,
                                                         spec.calculatorRoot,
                                                         spec.divingAgreementRoot,
                                                         nowMs());
    const fs::path exportRoot = spec.runtimeDir / "structured_exports" / sanitizeName(fs::path(spec.modelPath).stem().string());
    const json exportBundle = gguf_tensor_parser::buildStructuredExportBundle(spec.provider,
                                                                              spec.modelPath,
                                                                              inspection,
                                                                              nowMs());
    std::string exportError;
    const json exportManifest = gguf_tensor_parser::writeStructuredExportFiles(exportBundle, exportRoot, &exportError);
    if (doc.contains("conversion") && doc["conversion"].is_object()) {
        doc["conversion"]["gradientFit"] = exportBundle.value("fitResult", json::object());
        doc["conversion"]["structuredExport"] = exportManifest;
        if (!exportError.empty()) {
            doc["conversion"]["structuredExportError"] = exportError;
        }
        doc["conversion"]["personalityScheduler"] = {
            {"enabled", true},
            {"phaseBuckets", doc.value("model", json::object()).value("semanticBands", doc.value("conversion", json::object()).value("semanticBands", 6))},
            {"refreshEveryMs", 12000}
        };
    }
    writeJsonFile(outputPath, doc);
    return doc;
}

int effectiveCtxSizeForLaunch(const BackendRuntimeSpec &spec,
                              const gguf_tensor_parser::InspectResult &inspection) {
    int64_t effective = spec.ctxSize > 0 ? static_cast<int64_t>(spec.ctxSize) : 16384;
    if (spec.provider != "llamacpp") {
        return static_cast<int>(std::max<int64_t>(1, effective));
    }

    int64_t modelContextLength = 0;
    if (inspection.valid && inspection.report.is_object()) {
        const json model = inspection.report.value("model", json::object());
        modelContextLength = std::max<int64_t>(0, model.value("contextLength", static_cast<int64_t>(0)));
    }
    if (modelContextLength > 0) {
        effective = std::min<int64_t>(effective, modelContextLength);
    }

    const int64_t kvBytesPerToken = inferKvBytesPerToken(inspection);
    const uint64_t freeBytes = freePhysicalMemoryBytes();
    if (kvBytesPerToken > 0 && freeBytes > 0) {
        const uint64_t reserveBytes = 1024ull * 1024ull * 1024ull;
        const uint64_t kvBudgetBytes = freeBytes > reserveBytes ? (freeBytes - reserveBytes) : (freeBytes / 2);
        if (kvBudgetBytes > 0) {
            uint64_t memoryLimitedCtx = kvBudgetBytes / static_cast<uint64_t>(kvBytesPerToken);
            memoryLimitedCtx = roundDownPowerOfTwo(memoryLimitedCtx);
            if (memoryLimitedCtx > 0) {
                effective = std::min<int64_t>(effective, static_cast<int64_t>(std::max<uint64_t>(512, memoryLimitedCtx)));
            }
        }
    }

    if (effective <= 0) {
        effective = modelContextLength > 0 ? modelContextLength : 16384;
    }
    return static_cast<int>(std::max<int64_t>(1, effective));
}

} // namespace

json BackendRuntimeState::toJson() const {
    return {
        {"ready", ready},
        {"launchAttempted", launchAttempted},
        {"binaryFound", binaryFound},
        {"brainMapReady", brainMapReady},
        {"pid", pid},
        {"checkedAtMs", checkedAtMs},
        {"status", status},
        {"error", error},
        {"commandLine", commandLine},
        {"binaryPath", binaryPath.string()},
        {"statusFile", statusFile.string()},
        {"brainMapPath", brainMapPath.string()}
    };
}

json BugShooterState::toJson() const {
    return {
        {"enabled", enabled},
        {"running", running},
        {"executableFound", executableFound},
        {"launchAttempted", launchAttempted},
        {"pid", pid},
        {"checkedAtMs", checkedAtMs},
        {"status", status},
        {"error", error},
        {"statusFile", statusFile.string()}
    };
}

json readStatusFile(const fs::path &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return json::object();
    }
    json doc = json::parse(in, nullptr, false);
    if (doc.is_discarded()) {
        return json::object();
    }
    return doc;
}

bool ensureBackendReady(const BackendRuntimeSpec &spec, BackendRuntimeState &state) {
    state.checkedAtMs = nowMs();
    state.statusFile = spec.runtimeDir / (sanitizeName(spec.provider) + "_runtime_status.json");
    state.brainMapPath = spec.runtimeDir / "brain_maps" / (sanitizeName(fs::path(spec.modelPath).stem().string()) + ".json");
    gguf_tensor_parser::InspectResult inspection;
    buildBrainMap(spec, state.brainMapPath, &inspection);
    state.brainMapReady = fs::exists(state.brainMapPath);

    BackendRuntimeSpec launchSpec = spec;
    launchSpec.ctxSize = effectiveCtxSizeForLaunch(spec, inspection);

    const bool preferAdapterLauncher = hasAdapterLauncher(spec) && (spec.provider == "bitnet");
    if (preferAdapterLauncher) {
        state.binaryPath = fs::absolute(spec.pythonExecutable);
        state.binaryFound = true;
    } else {
        state.binaryPath = resolveBinary(spec);
        state.binaryFound = !state.binaryPath.empty();
    }

    std::string host;
    int port = 0;
    if (!parseHttpUrl(spec.baseUrl, host, port)) {
        state.ready = false;
        state.status = "invalid_base_url";
        state.error = "invalid base url: " + spec.baseUrl;
        writeJsonFile(state.statusFile, state.toJson());
        return false;
    }

    auto waitForReady = [&](int deadlineMs, int sleepMs) -> bool {
        int waited = 0;
        while (waited < deadlineMs) {
            if (backendReady(spec, host, port, std::min(400, sleepMs))) {
                return true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(sleepMs));
            waited += sleepMs;
        }
        return false;
    };

    if (backendReady(spec, host, port, 400)) {
        state.ready = true;
        state.status = "ready";
        state.error.clear();
        writeJsonFile(state.statusFile, state.toJson());
        return true;
    }

    const int deadlineMs = std::max(1000, spec.readyTimeoutMs);
    const int sleepMs = std::max(100, spec.healthPollMs);
    if (tcpReady(host, port, 400)) {
        if (waitForReady(deadlineMs, sleepMs)) {
            state.ready = true;
            state.status = "ready";
            state.error.clear();
            writeJsonFile(state.statusFile, state.toJson());
            return true;
        }
        state.ready = false;
        state.status = "health_timeout";
        state.error = spec.provider + " endpoint is listening but did not become model-ready";
        writeJsonFile(state.statusFile, state.toJson());
        return false;
    }

    bool usingAdapterLauncher = preferAdapterLauncher;
    if (!state.binaryFound && hasAdapterLauncher(spec)) {
        state.binaryPath = fs::absolute(spec.pythonExecutable);
        state.binaryFound = true;
        usingAdapterLauncher = true;
    }
    if (!state.binaryFound) {
        state.ready = false;
        state.status = "binary_missing";
        state.error = spec.provider + " adapter binary not found under runtime_store or provider root";
        writeJsonFile(state.statusFile, state.toJson());
        return false;
    }
    if (!spec.autoLaunch) {
        state.ready = false;
        state.status = "waiting_manual_launch";
        state.error = spec.provider + " endpoint offline and auto launch disabled";
        writeJsonFile(state.statusFile, state.toJson());
        return false;
    }

    state.launchAttempted = true;
    if (usingAdapterLauncher) {
        state.commandLine = buildAdapterCommandLine(launchSpec, host, port, state.brainMapPath);
    } else {
        state.commandLine = '"' + state.binaryPath.string() + '"' + ' ' + substituteTemplate(launchSpec.launchArgsTemplate, launchSpec.modelPath, host, port, state.brainMapPath, launchSpec);
    }
    state.pid = launchDetachedProcess(state.commandLine, state.binaryPath.parent_path());
    if (!state.pid) {
        state.ready = false;
        state.status = "launch_failed";
        state.error = "failed to spawn backend process";
        writeJsonFile(state.statusFile, state.toJson());
        return false;
    }

    if (waitForReady(deadlineMs, sleepMs)) {
        state.ready = true;
        state.status = "ready";
        state.error.clear();
        writeJsonFile(state.statusFile, state.toJson());
        return true;
    }

    state.ready = false;
    state.status = "health_timeout";
    state.error = spec.provider + " spawned but endpoint did not become model-ready";
    writeJsonFile(state.statusFile, state.toJson());
    return false;
}

bool ensureBugShooterAttached(const BugShooterSpec &spec, BugShooterState &state) {
    state.enabled = spec.enabled;
    state.checkedAtMs = nowMs();
    state.statusFile = spec.runtimeDir / "bug_shooter_status.json";
    if (!spec.enabled) {
        state.running = false;
        state.status = "disabled";
        state.error.clear();
        writeJsonFile(state.statusFile, state.toJson());
        return false;
    }
    std::error_code ec;
    state.executableFound = fs::exists(spec.executablePath, ec) && fs::is_regular_file(spec.executablePath, ec);
    if (!state.executableFound) {
        state.running = false;
        state.status = "binary_missing";
        state.error = "bug_shooter executable not found";
        writeJsonFile(state.statusFile, state.toJson());
        return false;
    }

    const json currentStatus = readStatusFile(state.statusFile);
    if (currentStatus.is_object() && currentStatus.value("running", false)) {
        state.running = true;
        state.pid = currentStatus.value("pid", 0u);
        state.status = currentStatus.value("status", std::string("running"));
        state.error.clear();
        return true;
    }

    state.launchAttempted = true;
    std::ostringstream cmd;
    cmd << '"' << spec.executablePath.string() << '"'
        << " --target-pid " << spec.targetPid
        << " --target-name \"" << spec.targetName << "\""
        << " --soft-limit-mb " << spec.softLimitMb
        << " --hard-limit-mb " << spec.hardLimitMb
        << " --poll-ms " << spec.pollIntervalMs
        << " --status-file \"" << state.statusFile.string() << "\"";
    state.pid = launchDetachedProcess(cmd.str(), spec.executablePath.parent_path());
    if (!state.pid) {
        state.running = false;
        state.status = "launch_failed";
        state.error = "failed to spawn bug_shooter";
        writeJsonFile(state.statusFile, state.toJson());
        return false;
    }
    state.running = true;
    state.status = "attached";
    state.error.clear();
    writeJsonFile(state.statusFile, state.toJson());
    return true;
}

} // namespace external_runtime