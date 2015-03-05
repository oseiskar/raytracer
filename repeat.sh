#!/bin/sh
set -e

rm -f out.raw.npy
python clray.py "$@"
while :
do
	python clray.py -a "$@"
done
