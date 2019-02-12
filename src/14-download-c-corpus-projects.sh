#!/bin/bash

set -eu -o pipefail

function download_archive {
    local url=$1
    local output_file=$2

    if [ ! -e data/raw/c-corpus-projects/${output_file} ]; then
        echo "downloading ${url}";
        curl -o data/raw/c-corpus-projects/${output_file} ${url};
    else
        echo "skipping ${url}, downloaded already"
    fi
}

function download_github_snapshot {
    local owner=$1
    local project=$2
    local tag=$3
    local output_file=$4

    download_archive https://codeload.github.com/${owner}/${project}/zip/${tag} ${output_file}.zip
}

mkdir -p data/raw/c-corpus-projects
download_github_snapshot torvalds linux v4.19 linux
download_github_snapshot chromium chromium 72.0.3610.1 chromium
download_github_snapshot openssl openssl OpenSSL_1_1_1 openssl
download_github_snapshot reactos reactos 9491979ac3fd9f4e3ec3898a43960e940ec2f198 reactos
download_github_snapshot git git v2.19.1 git
download_github_snapshot php php-src php-7.3.0beta3 php
download_github_snapshot postgres postgres REL_11_1 postgres
download_github_snapshot freebsd freebsd 65fbea0915b5a10a7e962afc79161dec4d9e4cd4 freebsd
download_github_snapshot gcc-mirror gcc gcc-8_2_0-release gcc
download_github_snapshot Rockbox rockbox 03718bdb76a3d9dd9a28caf862d590e78a6739aa rockbox
download_github_snapshot asterisk asterisk certified/13.21-cert3 asterisk
download_github_snapshot apache httpd 6ec1b34d040f3d629c2ae0ef9bb408abef8a89b7 apache
download_github_snapshot svn2github sdcc 5c4ba20deb6f403dc661ece4ac2b81092100e42a sdcc
download_github_snapshot Gnucash gnucash 3.3 gnucash
download_github_snapshot RTEMS rtems 841b54ee6a0669615aa55e3f94b2f72266f87a5f rtems
download_archive https://files.freeswitch.org/freeswitch-releases/freeswitch-1.8.2.zip freeswitch.zip
download_archive https://ffmpeg.org/releases/ffmpeg-4.1.tar.gz ffmpeg.tar.gz
download_archive http://ftp.gnu.org/gnu/glibc/glibc-2.28.tar.gz glibc.tar.gz
touch data/raw/c-corpus-projects
