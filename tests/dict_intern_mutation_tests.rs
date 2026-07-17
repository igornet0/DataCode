//! Dict stores must not mutate interned scalar cells shared with literals/constants.

#[cfg(test)]
mod tests {
    use data_code::{run, Value};

    fn assert_ok(source: &str) {
        run(source).expect("expected ok");
    }

    #[test]
    fn dict_assign_literal_one_then_increment_keys_independently() {
        assert_ok(
            r#"
freq = {}
freq["a"] = 1
freq["b"] = 1
freq["a"] += 1
if freq["a"] != 2 { throw ValueError("a") }
if freq["b"] != 1 { throw ValueError("b") }
if 4 > 1 { } else { throw ValueError("cmp") }
"#,
        );
    }

    #[test]
    fn huffman_freq_count_and_encode() {
        let result = run(
            r#"
import heapq

fn build_huffman_codes(freq) {
    if len(freq) == 0: return {}
    if len(freq) == 1 { return {freq.keys[0]: "0"} }
    heap = []
    uid = 0
    for ch in freq.keys {
        heapq.heappush(heap, (freq[ch], uid, ch, null, null))
        uid += 1
    }
    while len(heap) > 1 {
        a = heapq.heappop(heap)
        b = heapq.heappop(heap)
        merged = (a[0] + b[0], uid, null, a, b)
        uid += 1
        heapq.heappush(heap, merged)
    }
    root = heapq.heappop(heap)
    codes = {}
    fn walk(node, prefix) {
        ch = node[2]
        left = node[3]
        right = node[4]
        if ch != null and left == null and right == null {
            codes[ch] = if len(prefix) > 0 { prefix } else { "0" }
            return
        }
        if left != null: walk(left, prefix + "0")
        if right != null: walk(right, prefix + "1")
    }
    walk(root, "")
    return codes
}

fn huffman_encode(text, codes) {
    result = ""
    for ch in text { result += codes[ch] }
    return result
}

text = "aabbbcdddd"
freq = {}
for ch in text {
    if ch in freq { freq[ch] += 1 } else { freq[ch] = 1 }
}
codes = build_huffman_codes(freq)
huffman_encode(text, codes)
"#,
        );
        match result {
            Ok(Value::String(s)) => assert!(!s.is_empty()),
            Ok(v) => panic!("expected non-empty string, got {:?}", v),
            Err(e) => panic!("{:?}", e),
        }
    }
}
