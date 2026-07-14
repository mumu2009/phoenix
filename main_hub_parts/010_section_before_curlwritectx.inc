/* Auto-generated from main.cpp
   Segment: section_before_CurlWriteCtx
   Original lines: 2101-2109
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

// Disabled due to pre-existing compilation errors unrelated to LoRA integration
#if 0

/* CurlWriteCtx describes libcurl write callback output context */
// 调用方式：创建后作为 userdata 传给 CURLOPT_WRITEFUNCTION。
// 实现思路：保存目标字符串指针与最大字节上限。
// 注意事项：out 指针必须在回调生命周期内保持有效。
// 注意事项：maxBytes 为 0 时表示不限制回写长度。
// 注意事项：超过上限会主动截断并通过回调返回中止。
#ifdef HAVE_CURL
#endif
