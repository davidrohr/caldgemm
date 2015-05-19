#!/bin/bash
for i in "ulimit -m" "ulimit -v" "ulimit -l" "clinfo" "dmesg" "cat /var/log/messages"; do
    echo $i
    $i | tail -n 1000
done
