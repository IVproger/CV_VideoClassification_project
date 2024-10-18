# Data

Execute all commands inside `data` folder.

Requirements:

```bash
brew install wget rar
```

## Download data


```bash
wget --no-check-certificate -P tmp "https://www.crcv.ucf.edu/data/UCF50.rar"
```

OR

```bash
bash scripts/download.sh
```


## Unarchive data

```bash
mkdir -p raw raw/UCF50
unrar x tmp/UCF50.rar raw
```

OR

```bash
bash scripts/unarchive.sh
```

## Structure

```
data/
- raw/
- - UCF50/
- - - <Class1>
- - - - *.avi
- - - <Class2>
- - - - *.avi
- - - ...
- scripts/
- - *.sh
- - *.py
- tmp/ (installers, archives, etc.)
```
