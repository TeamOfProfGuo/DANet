echo "Your environment is: $1"
rm -r ~/opt/anaconda3/envs/$1/lib/python3.6/site-packages/encoding/
cp -r ./encoding ~/opt/anaconda3/envs/$1/lib/python3.6/site-packages/encoding/
echo "DONE"

