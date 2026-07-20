import pathlib

p = pathlib.Path("compile_gtest.bat")
s = p.read_text(encoding="utf-8").replace("\r\n", "\n")

old = '"%GXX_EXE%" -o gtest_runner.exe -std=c++20 -Wa,-mbig-obj -DAI_EXTERNAL_BACKEND_COMPAT=1 -DAI_EXTERNAL_LEARNER_BRIDGE=1 -DHAVE_SQLITE @"%CONAN_CFLAGS_FILE%" -I"%CD%" -I"%CD%\\poppler-25.12.0\\Library\\include" -I"%PY_INC%" %OUTSIDES_CFLAGS% -I"%CD%\\tests\\gtest" %GTEST_SOURCES% %COMMON_SOURCES% -Wl,--start-group @"%CONAN_LIBS_FILE%" -Wl,--end-group "%CD%\\poppler-25.12.0\\Library\\lib\\poppler-cpp.lib" "%CD%\\poppler-25.12.0\\Library\\lib\\poppler.lib" -L"%PY_LIB%" -l%PY_LINK_NAME% -lws2_32 -O3 -g'
new = '"%GXX_EXE%" -o gtest_runner.exe -std=c++20 -Wa,-mbig-obj -DAI_EXTERNAL_BACKEND_COMPAT=1 -DAI_EXTERNAL_LEARNER_BRIDGE=1 -DHAVE_SQLITE @"%CONAN_CFLAGS_FILE%" -I"%CD%" -I"%CD%\\poppler-25.12.0\\Library\\include" -I"%PY_INC%" %OUTSIDES_CFLAGS% -I"%CD%\\tests\\gtest" %GTEST_SOURCES% %COMMON_SOURCES% -Wl,--start-group @"%CONAN_LIBS_FILE%" -Wl,--end-group "%CD%\\poppler-25.12.0\\Library\\lib\\poppler-cpp.lib" "%CD%\\poppler-25.12.0\\Library\\lib\\poppler.lib" -L"%PY_LIB%" -l%PY_LINK_NAME% -lws2_32 -O0 -g'

if old not in s:
    print("old not found")
    raise SystemExit(1)

s = s.replace(old, new, 1)
p.write_text(s, encoding="utf-8", newline="\n")
print("patched compile_gtest.bat to -O0")
