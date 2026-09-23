#pragma once

#include "runtime_opt/platform.hpp"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace phoenix {
namespace runtime_opt {

struct ShmConfig {
  TriSwitch enabled = TriSwitch::Auto;
  std::string name = "phoenix_runtime_opt";
  size_t bytes = 65536;
  std::string fallback = "file";
};

inline bool shmAutoOn(HostPlatform) { return true; }

class ShmChannel {
public:
  ~ShmChannel() { close(); }

  bool open(const ShmConfig &cfg, const std::filesystem::path &fallbackDir,
            bool forceFallback = false) {
    close();
    cfg_ = cfg;
    if (cfg.bytes < 64)
      return false;
    if (!forceFallback && tryOpenShm(cfg)) {
      backend_ = "shm";
      return true;
    }
    if (cfg.fallback == "none")
      return false;
    return openFile(cfg, fallbackDir);
  }

  const std::string &backend() const { return backend_; }

  bool write(std::string_view payload) {
    if (!mapped_ || mappedSize_ < 16)
      return false;
    const uint32_t n = static_cast<uint32_t>(
        std::min(payload.size(), static_cast<size_t>(mappedSize_ - 16)));
    std::memcpy(mapped_, "PHXOPT1", 8);
    std::memcpy(static_cast<char *>(mapped_) + 8, &n, 4);
    std::memcpy(static_cast<char *>(mapped_) + 12, payload.data(), n);
    if (backend_ == "file" && !filePath_.empty()) {
      std::ofstream out(filePath_, std::ios::binary | std::ios::trunc);
      if (!out)
        return false;
      out.write(static_cast<const char *>(mapped_), static_cast<std::streamsize>(16 + n));
    }
    return true;
  }

  bool read(std::string &out) const {
    if (!mapped_ || mappedSize_ < 16)
      return false;
    if (std::memcmp(mapped_, "PHXOPT1", 8) != 0)
      return false;
    uint32_t n = 0;
    std::memcpy(&n, static_cast<const char *>(mapped_) + 8, 4);
    if (n > mappedSize_ - 16)
      return false;
    out.assign(static_cast<const char *>(mapped_) + 12, n);
    return true;
  }

  void close() {
#ifdef _WIN32
    if (backend_ == "shm" && mapped_) {
      UnmapViewOfFile(mapped_);
    }
    mapped_ = nullptr;
    if (mapHandle_) {
      CloseHandle(mapHandle_);
      mapHandle_ = nullptr;
    }
#else
    if (backend_ == "shm" && mapped_ && mapped_ != MAP_FAILED && mappedSize_) {
      munmap(mapped_, mappedSize_);
    }
    mapped_ = nullptr;
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
    if (!posixName_.empty()) {
      /* leave named shm for peers; unlink only if we created in tests via force */
    }
#endif
    fileBuf_.clear();
    mappedSize_ = 0;
    backend_.clear();
  }

private:
  bool tryOpenShm(const ShmConfig &cfg) {
#ifdef _WIN32
    std::string winName = "Local\\" + cfg.name;
    mapHandle_ = CreateFileMappingA(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE,
                                    0, static_cast<DWORD>(cfg.bytes), winName.c_str());
    if (!mapHandle_)
      return false;
    mapped_ = MapViewOfFile(mapHandle_, FILE_MAP_ALL_ACCESS, 0, 0, cfg.bytes);
    if (!mapped_) {
      CloseHandle(mapHandle_);
      mapHandle_ = nullptr;
      return false;
    }
    mappedSize_ = cfg.bytes;
    return true;
#else
    posixName_ = cfg.name;
    if (posixName_.empty() || posixName_[0] != '/')
      posixName_ = "/" + cfg.name;
    fd_ = shm_open(posixName_.c_str(), O_RDWR | O_CREAT, 0600);
    if (fd_ < 0)
      return false;
    if (ftruncate(fd_, static_cast<off_t>(cfg.bytes)) != 0) {
      ::close(fd_);
      fd_ = -1;
      return false;
    }
    mapped_ = mmap(nullptr, cfg.bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (mapped_ == MAP_FAILED) {
      mapped_ = nullptr;
      ::close(fd_);
      fd_ = -1;
      return false;
    }
    mappedSize_ = cfg.bytes;
    return true;
#endif
  }

  bool openFile(const ShmConfig &cfg, const std::filesystem::path &fallbackDir) {
    std::error_code ec;
    std::filesystem::create_directories(fallbackDir, ec);
    filePath_ = fallbackDir / (cfg.name + ".bin");
    fileBuf_.assign(cfg.bytes, 0);
    if (std::filesystem::exists(filePath_)) {
      std::ifstream in(filePath_, std::ios::binary);
      in.read(fileBuf_.data(), static_cast<std::streamsize>(cfg.bytes));
    }
    mapped_ = fileBuf_.data();
    mappedSize_ = fileBuf_.size();
    backend_ = "file";
    return true;
  }

  ShmConfig cfg_{};
  std::string backend_;
  std::filesystem::path filePath_;
  std::vector<char> fileBuf_;
  void *mapped_ = nullptr;
  size_t mappedSize_ = 0;
#ifdef _WIN32
  HANDLE mapHandle_ = nullptr;
#else
  int fd_ = -1;
  std::string posixName_;
#endif
};

} // namespace runtime_opt
} // namespace phoenix
