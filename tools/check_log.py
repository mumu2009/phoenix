import pathlib, sys, glob
for p in sorted(glob.glob('build/tmp/compile_gtest_run*.log')):
    try:
        st = pathlib.Path(p).stat()
        print(p, 'size=', st.st_size, 'mtime=', st.st_mtime)
    except Exception as e:
        print(p, 'error', e)
