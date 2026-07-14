/* download_gguf_model.cpp - GGUF model download tool
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

#include <algorithm>
#include <windows.h>
#include <winhttp.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#ifdef _MSC_VER
#pragma comment(lib, "winhttp.lib")
#endif

namespace
{
namespace fs = std::filesystem;

struct Options
{
    std::wstring url;
    fs::path outputDir = L"GGUF_models";
    std::optional<std::wstring> fileName;
    std::optional<std::wstring> bearerToken;
    bool forceOverwrite{false};
    int timeoutSeconds{600};
};

std::wstring toWide(const std::string &value)
{
    if (value.empty())
        return {};
    int size = MultiByteToWideChar(CP_UTF8, 0, value.c_str(), static_cast<int>(value.size()), nullptr, 0);
    if (size <= 0)
        return {};
    std::wstring out(static_cast<size_t>(size), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, value.c_str(), static_cast<int>(value.size()), out.data(), size);
    return out;
}

std::wstring trim(const std::wstring &value)
{
    size_t first = value.find_first_not_of(L" \t\r\n");
    if (first == std::wstring::npos)
        return L"";
    size_t last = value.find_last_not_of(L" \t\r\n");
    return value.substr(first, last - first + 1);
}

void printUsage()
{
    std::wcerr
        << L"Usage:\n"
        << L"  download_gguf_model.exe --url <model-url> [--name <file.gguf>] [--dir <GGUF_models>] [--token <bearer>] [--force] [--timeout <seconds>]\n\n"
        << L"Examples:\n"
        << L"  download_gguf_model.exe --url https://example.com/model.gguf\n"
        << L"  download_gguf_model.exe --url https://huggingface.co/.../resolve/main/model.gguf --name qwen.gguf --token %HF_TOKEN%\n";
}

bool parseArgs(int argc, char **argv, Options &options, std::wstring &error)
{
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        auto requireValue = [&](std::string_view name) -> std::optional<std::string> {
            if (i + 1 >= argc)
            {
                error = toWide(std::string("missing value for ") + std::string(name));
                return std::nullopt;
            }
            return std::string(argv[++i]);
        };

        if (arg == "--url")
        {
            auto value = requireValue("--url");
            if (!value)
                return false;
            options.url = toWide(*value);
        }
        else if (arg == "--name")
        {
            auto value = requireValue("--name");
            if (!value)
                return false;
            options.fileName = toWide(*value);
        }
        else if (arg == "--dir")
        {
            auto value = requireValue("--dir");
            if (!value)
                return false;
            options.outputDir = fs::path(toWide(*value));
        }
        else if (arg == "--token")
        {
            auto value = requireValue("--token");
            if (!value)
                return false;
            options.bearerToken = toWide(*value);
        }
        else if (arg == "--timeout")
        {
            auto value = requireValue("--timeout");
            if (!value)
                return false;
            try
            {
                options.timeoutSeconds = std::max(1, std::stoi(*value));
            }
            catch (...)
            {
                error = L"invalid integer for --timeout";
                return false;
            }
        }
        else if (arg == "--force")
        {
            options.forceOverwrite = true;
        }
        else if (arg == "--help" || arg == "-h")
        {
            printUsage();
            std::exit(0);
        }
        else
        {
            error = toWide(std::string("unknown argument: ") + arg);
            return false;
        }
    }

    if (options.url.empty())
    {
        error = L"--url is required";
        return false;
    }
    return true;
}

std::wstring deriveFileNameFromUrl(const std::wstring &url)
{
    std::wstring clean = url;
    size_t queryPos = clean.find(L'?');
    if (queryPos != std::wstring::npos)
        clean.resize(queryPos);
    size_t fragmentPos = clean.find(L'#');
    if (fragmentPos != std::wstring::npos)
        clean.resize(fragmentPos);
    size_t slashPos = clean.find_last_of(L"/");
    if (slashPos == std::wstring::npos || slashPos + 1 >= clean.size())
        return L"downloaded_model.gguf";
    std::wstring name = clean.substr(slashPos + 1);
    return name.empty() ? L"downloaded_model.gguf" : name;
}

std::wstring humanSize(std::uint64_t bytes)
{
    static const wchar_t *units[] = {L"B", L"KB", L"MB", L"GB", L"TB"};
    double value = static_cast<double>(bytes);
    size_t unit = 0;
    while (value >= 1024.0 && unit < 4)
    {
        value /= 1024.0;
        ++unit;
    }
    std::wostringstream oss;
    oss << std::fixed << std::setprecision(unit == 0 ? 0 : 2) << value << L' ' << units[unit];
    return oss.str();
}

std::wstring formatLastError(DWORD code)
{
    LPWSTR buffer = nullptr;
    DWORD len = FormatMessageW(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr,
        code,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        reinterpret_cast<LPWSTR>(&buffer),
        0,
        nullptr);
    std::wstring message = (len > 0 && buffer) ? trim(buffer) : (L"Win32 error " + std::to_wstring(code));
    if (buffer)
        LocalFree(buffer);
    return message;
}

class WinHttpHandle
{
public:
    WinHttpHandle() = default;
    explicit WinHttpHandle(HINTERNET handle) : handle_(handle) {}
    ~WinHttpHandle()
    {
        if (handle_)
            WinHttpCloseHandle(handle_);
    }

    WinHttpHandle(const WinHttpHandle &) = delete;
    WinHttpHandle &operator=(const WinHttpHandle &) = delete;

    WinHttpHandle(WinHttpHandle &&other) noexcept : handle_(other.handle_)
    {
        other.handle_ = nullptr;
    }

    WinHttpHandle &operator=(WinHttpHandle &&other) noexcept
    {
        if (this != &other)
        {
            if (handle_)
                WinHttpCloseHandle(handle_);
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    HINTERNET get() const { return handle_; }
    explicit operator bool() const { return handle_ != nullptr; }

private:
    HINTERNET handle_{nullptr};
};

bool queryContentLength(HINTERNET request, std::uint64_t &value)
{
    wchar_t buffer[64] = {};
    DWORD size = sizeof(buffer);
    if (!WinHttpQueryHeaders(request, WINHTTP_QUERY_CONTENT_LENGTH, WINHTTP_HEADER_NAME_BY_INDEX, buffer, &size, WINHTTP_NO_HEADER_INDEX))
        return false;
    try
    {
        value = std::stoull(buffer);
        return true;
    }
    catch (...)
    {
        return false;
    }
}

bool queryStatusCode(HINTERNET request, DWORD &statusCode)
{
    DWORD size = sizeof(statusCode);
    return WinHttpQueryHeaders(request, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER, WINHTTP_HEADER_NAME_BY_INDEX, &statusCode, &size, WINHTTP_NO_HEADER_INDEX) == TRUE;
}

bool downloadFile(const Options &options, std::wstring &message)
{
    URL_COMPONENTS parts{};
    parts.dwStructSize = sizeof(parts);
    parts.dwSchemeLength = static_cast<DWORD>(-1);
    parts.dwHostNameLength = static_cast<DWORD>(-1);
    parts.dwUrlPathLength = static_cast<DWORD>(-1);
    parts.dwExtraInfoLength = static_cast<DWORD>(-1);

    std::vector<wchar_t> urlBuffer(options.url.begin(), options.url.end());
    urlBuffer.push_back(L'\0');
    if (!WinHttpCrackUrl(urlBuffer.data(), 0, 0, &parts))
    {
        message = L"invalid url: " + formatLastError(GetLastError());
        return false;
    }

    std::wstring host(parts.lpszHostName, parts.dwHostNameLength);
    std::wstring path(parts.lpszUrlPath, parts.dwUrlPathLength);
    if (parts.dwExtraInfoLength > 0 && parts.lpszExtraInfo)
        path.append(parts.lpszExtraInfo, parts.dwExtraInfoLength);

    fs::path finalDir = fs::absolute(options.outputDir);
    fs::create_directories(finalDir);

    std::wstring fileName = options.fileName.value_or(deriveFileNameFromUrl(options.url));
    fs::path finalPath = finalDir / fs::path(fileName);
    fs::path tempPath = finalPath;
    tempPath += L".part";

    if (fs::exists(finalPath) && !options.forceOverwrite)
    {
        message = L"target already exists, use --force to overwrite: " + finalPath.wstring();
        return false;
    }

    WinHttpHandle session(WinHttpOpen(L"v6.0Alixander-GGUF-Downloader/1.0",
                                      WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
                                      WINHTTP_NO_PROXY_NAME,
                                      WINHTTP_NO_PROXY_BYPASS,
                                      0));
    if (!session)
    {
        message = L"WinHttpOpen failed: " + formatLastError(GetLastError());
        return false;
    }

    int timeoutMs = options.timeoutSeconds * 1000;
    WinHttpSetTimeouts(session.get(), timeoutMs, timeoutMs, timeoutMs, timeoutMs);

    WinHttpHandle connection(WinHttpConnect(session.get(), host.c_str(), parts.nPort, 0));
    if (!connection)
    {
        message = L"WinHttpConnect failed: " + formatLastError(GetLastError());
        return false;
    }

    DWORD flags = (parts.nScheme == INTERNET_SCHEME_HTTPS) ? WINHTTP_FLAG_SECURE : 0;
    WinHttpHandle request(WinHttpOpenRequest(connection.get(),
                                             L"GET",
                                             path.c_str(),
                                             nullptr,
                                             WINHTTP_NO_REFERER,
                                             WINHTTP_DEFAULT_ACCEPT_TYPES,
                                             flags));
    if (!request)
    {
        message = L"WinHttpOpenRequest failed: " + formatLastError(GetLastError());
        return false;
    }

    std::wstring headers;
    if (options.bearerToken && !options.bearerToken->empty())
        headers = L"Authorization: Bearer " + *options.bearerToken + L"\r\n";

    BOOL sendOk = WinHttpSendRequest(request.get(),
                                     headers.empty() ? WINHTTP_NO_ADDITIONAL_HEADERS : headers.c_str(),
                                     headers.empty() ? 0 : static_cast<DWORD>(headers.size()),
                                     WINHTTP_NO_REQUEST_DATA,
                                     0,
                                     0,
                                     0);
    if (!sendOk)
    {
        message = L"WinHttpSendRequest failed: " + formatLastError(GetLastError());
        return false;
    }
    if (!WinHttpReceiveResponse(request.get(), nullptr))
    {
        message = L"WinHttpReceiveResponse failed: " + formatLastError(GetLastError());
        return false;
    }

    DWORD statusCode = 0;
    if (!queryStatusCode(request.get(), statusCode))
    {
        message = L"unable to query HTTP status: " + formatLastError(GetLastError());
        return false;
    }
    if (statusCode < 200 || statusCode >= 300)
    {
        message = L"download failed with HTTP status " + std::to_wstring(statusCode);
        return false;
    }

    std::uint64_t totalBytes = 0;
    bool hasTotalBytes = queryContentLength(request.get(), totalBytes);

    std::ofstream output(tempPath, std::ios::binary | std::ios::trunc);
    if (!output)
    {
        message = L"unable to open output file: " + tempPath.wstring();
        return false;
    }

    std::vector<char> buffer(1 << 16);
    std::uint64_t downloaded = 0;
    auto lastPrint = std::chrono::steady_clock::now() - std::chrono::seconds(1);

    while (true)
    {
        DWORD available = 0;
        if (!WinHttpQueryDataAvailable(request.get(), &available))
        {
            message = L"WinHttpQueryDataAvailable failed: " + formatLastError(GetLastError());
            output.close();
            fs::remove(tempPath);
            return false;
        }
        if (available == 0)
            break;

        while (available > 0)
        {
            DWORD chunk = static_cast<DWORD>(std::min<std::size_t>(buffer.size(), available));
            DWORD bytesRead = 0;
            if (!WinHttpReadData(request.get(), buffer.data(), chunk, &bytesRead))
            {
                message = L"WinHttpReadData failed: " + formatLastError(GetLastError());
                output.close();
                fs::remove(tempPath);
                return false;
            }
            if (bytesRead == 0)
                break;

            output.write(buffer.data(), static_cast<std::streamsize>(bytesRead));
            if (!output)
            {
                message = L"write failed for file: " + tempPath.wstring();
                output.close();
                fs::remove(tempPath);
                return false;
            }

            downloaded += bytesRead;
            available -= bytesRead;

            auto now = std::chrono::steady_clock::now();
            if (now - lastPrint >= std::chrono::milliseconds(250))
            {
                std::wcout << L"\r[DOWNLOADING] " << fileName << L"  " << humanSize(downloaded);
                if (hasTotalBytes && totalBytes > 0)
                {
                    double percent = (static_cast<double>(downloaded) * 100.0) / static_cast<double>(totalBytes);
                    std::wcout << L" / " << humanSize(totalBytes) << L"  " << std::fixed << std::setprecision(1) << percent << L"%";
                }
                std::wcout << L"      " << std::flush;
                lastPrint = now;
            }
        }
    }

    output.close();
    if (fs::exists(finalPath))
        fs::remove(finalPath);
    fs::rename(tempPath, finalPath);

    std::wcout << L"\r[DOWNLOADING] " << fileName << L"  " << humanSize(downloaded);
    if (hasTotalBytes && totalBytes > 0)
        std::wcout << L" / " << humanSize(totalBytes) << L"  100.0%";
    std::wcout << L"\n";

    message = L"download complete: " + finalPath.wstring();
    return true;
}
} // namespace

int main(int argc, char **argv)
{
    Options options;
    std::wstring error;
    if (!parseArgs(argc, argv, options, error))
    {
        std::wcerr << L"[ERROR] " << error << L"\n\n";
        printUsage();
        return 1;
    }

    try
    {
        std::wstring message;
        if (!downloadFile(options, message))
        {
            std::wcerr << L"[ERROR] " << message << L"\n";
            return 2;
        }
        std::wcout << L"[OK] " << message << L"\n";
        return 0;
    }
    catch (const std::exception &ex)
    {
        std::wcerr << L"[ERROR] " << toWide(ex.what()) << L"\n";
        return 3;
    }
}