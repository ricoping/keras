sentence=$1
mecab -b 81920 << EOF
$sentence
EOF

exit 0
