# addons

本目录存放 Phoenix 主程序内建插件模块，供主程序在启动或运行期注册使用。

## 目录内容

- `builtin_registry.cpp/.hpp/.py`：内建插件注册表，统一维护插件发现与绑定。
- `math_addon.cpp/.hpp/.py`：数学相关扩展能力。
- `search_addon.cpp/.hpp/.py`：检索与搜索相关扩展能力。
- `__init__.py`：Python 侧包入口。

## 作用

1. 为主程序提供可插拔扩展点。
2. 保持 C++ 与 Python 两侧的插件定义对齐。
3. 作为 `addon.cpp` 的实际扩展实现来源。

## 使用说明

1. 新增插件时，优先补齐 `.cpp/.hpp` 与对应 `.py` 适配层。
2. 在 `builtin_registry` 中完成注册，确保主程序能够发现。
3. 若要替换默认实现，结合 `module_overrides/README.md` 中的挂载机制使用。