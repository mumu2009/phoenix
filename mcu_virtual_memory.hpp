/* mcu_virtual_memory.hpp - MCU virtual memory subsystem */

#pragma once

/**
 * MCU virtual memory subsystem (v2.1)
 *
 * Supports bare metal MCU + external parallel SDRAM solution.
 * 7B GGUF model (~250MB) directly resides in external SDRAM (256MB/512MB), no paging needed.
 * SD card is only used as paging backup when SDRAM capacity is insufficient or running larger models.
 */

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <map>
#include <memory>

namespace mcu_vm {

/* Memory tiers (from fast to slow) */
enum class Tier {
    InternalSram,   /* Fastest, smallest capacity (KB level), for stack and hot code */
    ExternalSdram,  /* External parallel SDRAM, 256MB/512MB, 7B GGUF weights reside here */
    Psdram,         /* Off-chip PSRAM (small form factor), 8~64MB */
    Sdcard,         /* Cold storage, sequential read 20~50MB/s, random read slow; model cold-start load source */
    RemoteHost,     /* Network remote host (Wi-Fi download) */
};

struct TierInfo {
    Tier tier;                      /* Memory tier */
    std::size_t totalBytes{0};      /* Total bytes in tier */
    std::size_t freeBytes{0};       /* Free bytes in tier */
    std::size_t pageSize{4096};     /* Page size in bytes */
    uint32_t readLatencyUs{0};      /* Typical read latency in microseconds */
    uint32_t readThroughputKBps{0}; /* Sequential read throughput in KB/s */
};

/* Weight block (corresponds to GGUF tensor or its slice) */
struct WeightBlock {
    int blockId{0};                 /* Block identifier */
    std::string tensorName;         /* Tensor name */
    std::size_t offsetInFile{0};    /* Offset in file */
    std::size_t bytes{0};           /* Block size in bytes */
    Tier currentTier{Tier::ExternalSdram}; /* Current memory tier */
    uint32_t hitCount{0};           /* Access hit count */
    bool pinned{false};             /* Pinned flag (prevent eviction) */
    void *residentAddr{nullptr};    /* Address in SDRAM */
};

/* Virtual memory configuration */
struct Config {
    std::size_t sdramTotalBytes{512ULL << 20};    /* 512MB external SDRAM (32-bit EXMC) */
    std::size_t internalSramBytes{384 << 10};     /* 384KB internal SRAM (GD32H759) */
    std::size_t psramTotalBytes{0};              /* No PSRAM by default */
    std::size_t sdcardPageCacheBytes{32 << 20};   /* 32MB SD card cache (backup only) */
    std::size_t pageSize{4096};                 /* Page size in bytes */
    bool enableSdSwap{false};                    /* 7B GGUF resides in SDRAM, no paging by default */
    bool enableSdramResidency{true};             /* Weights directly reside in SDRAM */
    bool sdramAsHeap{true};                      /* malloc allocates from SDRAM by default */
    std::string sdCardWeightsRoot{"runtime_store/sdcard_weights"}; /* SD card weights root */
    std::string ggufModelPath{};                 /* GGUF model path */
};

/* Manager interface */
class Manager {
public:
    virtual ~Manager() = default;

    virtual bool init(const Config &cfg) = 0; /* Initialize manager */
    virtual void shutdown() = 0;              /* Shutdown manager */

    /* Register a weight block (usually called after parsing GGUF metadata) */
    virtual bool registerBlock(const WeightBlock &block) = 0;

    /* Get weight block address in local memory; page from SD card to external SDRAM if needed */
    virtual void *resolve(int blockId, std::size_t offset, std::size_t length) = 0;

    /* Hint that a weight block will be used soon, can prefetch */
    virtual void prefetch(int blockId) = 0;

    /* Statistics */
    virtual std::size_t residentBytes() const = 0; /* Resident bytes in SDRAM */
    virtual std::size_t swappedBytes() const = 0;  /* Swapped bytes to SD card */
    virtual uint64_t pageFaults() const = 0;       /* Page fault count */
    virtual uint64_t sdcardReads() const = 0;      /* SD card read count */

    /* Memory tier snapshot */
    virtual std::vector<TierInfo> tierSnapshot() const = 0;
};

/* Factory function: create appropriate Manager for target platform */
std::unique_ptr<Manager> createManagerForTarget(const std::string &targetSoc);

} // namespace mcu_vm
