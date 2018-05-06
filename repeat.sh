command=$@

eval $command
while [ $? -ne 1 ]; do
    echo 'ERROR during training, possible crash! Attempting to resume...'
    eval $command
done
