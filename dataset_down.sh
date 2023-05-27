#!/bin/bash
# if [ ! -f './gdown.pl' ]; then
#     wget https://raw.github.com/circulosmeos/gdown.pl/master/gdown.pl
#     chmod u+x gdown.pl
# fi

file='./test_scan.zip'
if [ -f $file ]; then
    echo $file+' exist'
else
    echo $file+' not exist'
    # gdown https://drive.google.com/uc?id=1uuuvlmPtrjGuG83tAzxMbHBqHSlFpGjr test_scan.zip
    wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1DEavUuFh6x7n1XEMQCaW8yaU3wQxqfBu' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1DEavUuFh6x7n1XEMQCaW8yaU3wQxqfBu" -O test_scan.zip && rm -rf ~/cookies.txt
    chmod 755 $file
    # gdown https://drive.google.com/uc?id=1zmfrXzT9lnLg7NlQ-hXekZlyX9aGNNqj
fi

file='./train_clean.zip'
if [ -f $file ]; then
    echo $file+' exist'
else
    echo $file+' not exist'
    # gdown https://drive.google.com/uc?id=18larBxdbutp0u1GCHo7QJ2qo4f48SU0C #cjh 드라이브
    wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=18larBxdbutp0u1GCHo7QJ2qo4f48SU0C' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=18larBxdbutp0u1GCHo7QJ2qo4f48SU0C" -O train_clean.zip && rm -rf ~/cookies.txt
    chmod 755 $file
    # gdown https://drive.google.com/uc?id=1cqxSVFxfonx5qKVIdfByOq7uYdNZY8ea
fi

file='./train_scan.zip'
if [ -f $file ]; then
    echo $file+' exist'
else
    echo $file+' not exist'
    # gdown https://drive.google.com/uc?id=17YqE_aLUEwwVgTSzoZ2Awy5A3A8BmssV
    wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=17YqE_aLUEwwVgTSzoZ2Awy5A3A8BmssV' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=17YqE_aLUEwwVgTSzoZ2Awy5A3A8BmssV" -O train_scan.zip && rm -rf ~/cookies.txt
    chmod 755 $file
    # gdown https://drive.google.com/uc?id=16zAeGDmqbAvn7Iy8V-mBylKx6rG-wgLD
fi

echo 'done!'
exit 0