#!/bin/bash

# make dirs if not exist
mkdir -p raw raw/UCF50

# unarchive
unrar x tmp/UCF50.rar raw -y
