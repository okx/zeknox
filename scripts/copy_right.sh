for file in $(find cuda -name "*.cpp" -o -name "*.h" -o -name "*.hpp"); do
    if ! grep -q "Copyright" "$file"; then
        echo "// Copyright 2024 OKX\n$(cat "$file")" > "$file"
    fi
done