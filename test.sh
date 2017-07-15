command='./train.py --config examples/cgan.config --dir workspace/cgan_test2'

eval $command
while [ $? -ne 1 ]; do
    echo 'ERROR during training, possible crash! Attempting to resume...'
    eval $command
done
