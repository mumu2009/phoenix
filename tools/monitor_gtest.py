import pathlib, time, os
p = pathlib.Path('build/tmp/gtest_full_run4.log')
for i in range(5):
    if p.exists():
        st = p.stat()
        print('size=', st.st_size, 'mtime=', st.st_mtime)
    else:
        print('missing')
    time.sleep(2)
