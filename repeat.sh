#!/bin/sh
set -e

rm -f out.raw.npy
python clray.py "$1"
while :
do
	python clray.py -a "$1"
done
