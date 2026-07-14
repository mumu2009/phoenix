/* Auto-generated from main.cpp
   Segment: section_before_ProcessReservedArena
   Original lines: 1650-1684
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

static constexpr int kSharedMemoryProtocolVersion = 2;

static std::string currentOutsidesLinkMode() { return AI_OUTSIDES_LINK_MODE; }

static json buildShmEnvelope(const json &payload, const std::string &controller,
                             const std::string &backend,
                             const std::string &phase) {
  return json{{"shmProtocolVersion", kSharedMemoryProtocolVersion},
              {"transport", "inproc-vector-shm2"},
              {"phase", phase},
              {"controller", controller},
              {"backend", backend},
              {"timestampMs", nowEpochMs()},
              {"payload", payload}};
}

static bool unpackShmEnvelope(const json &doc, json &payloadOut) {
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
