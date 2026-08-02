#!/usr/bin/env bash
# Environment setup for building Phoenix on the RDK X5.
# This script creates local pkg-config files for the Drogon/Trantor
# libraries installed under /usr/local (which do not ship .pc files),
# and points pkg-config at them.

root="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
pc_dir="$root/build/x5_pkgconfig"
mkdir -p "$pc_dir"

# Drogon pkg-config.  Depends on the Trantor, jsoncpp, OpenSSL, zlib,
# yaml-cpp, uuid and brotli libraries that are already present on the X5.
if [[ ! -f "$pc_dir/drogon.pc" ]]; then
cat > "$pc_dir/drogon.pc" <<'EOF'
prefix=/usr/local
libdir=${prefix}/lib
includedir=${prefix}/include

Name: drogon
Description: Drogon C++ web framework
Version: 1.9.5
Libs: -L${libdir} -ldrogon -ldl
Cflags: -I${includedir} -DHAS_YAML_CPP
Requires: trantor jsoncpp openssl zlib yaml-cpp uuid libbrotlidec libbrotlienc libbrotlicommon
EOF
fi

# Trantor pkg-config.  Depends on OpenSSL and pthreads.
if [[ ! -f "$pc_dir/trantor.pc" ]]; then
cat > "$pc_dir/trantor.pc" <<'EOF'
prefix=/usr/local
libdir=${prefix}/lib
includedir=${prefix}/include

Name: trantor
Description: Trantor C++ non-blocking network library
Version: 1.5.19
Libs: -L${libdir} -ltrantor
Cflags: -I${includedir}
Requires: openssl
EOF
fi

export PKG_CONFIG_PATH="$pc_dir${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
