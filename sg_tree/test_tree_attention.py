"""
Tree Attention API 测试脚本

用法:
    python test_tree_attention.py --server http://localhost:8000

测试内容:
    1. 普通请求正常工作
    2. 简单树 (1 个分支)
    3. 多分支树 (2 个分支)
    4. 深层树 (分支上的分支)
    5. 较长序列树
"""

import argparse
import requests
import sys
import time


def generate(server, **kwargs):
    resp = requests.post(f"{server}/generate", json=kwargs, timeout=60)
    assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.text[:500]}"
    return resp.json()


def get_logprobs(result):
    return result["meta_info"]["input_token_logprobs"]


def check(name, actual, expected, tol=0.001):
    if actual is None and expected is None:
        return True
    if actual is None or expected is None:
        print(f"    FAIL {name}: actual={actual}, expected={expected}")
        return False
    diff = abs(actual - expected)
    if diff <= tol:
        return True
    else:
        print(f"    FAIL {name}: actual={actual:.6f}, expected={expected:.6f}, diff={diff:.6f}")
        return False


def test_normal_request(server):
    """测试普通请求是否正常"""
    print("Test 1: 普通文本请求")
    result = generate(
        server,
        text="The quick brown fox jumps over the lazy dog",
        sampling_params={"max_new_tokens": 0},
        return_logprob=True,
        logprob_start_len=0,
        top_logprobs_num=5,
    )
    lps = get_logprobs(result)
    assert len(lps) > 0, "No logprobs returned"
    print(f"    返回 {len(lps)} 个 token 的 logprob - PASS")
    return True


def test_simple_tree(server):
    """简单树: 主序列 + 1 个替代 token"""
    print("\nTest 2: 简单树 (5 tokens + 1 alt)")
    #
    # 树结构:
    #   0 -> 1 -> 2 -> 3 -> 4
    #             \-> 5 (alt, parent=1)
    #
    main_ids = [9707, 1879, 11, 419, 374]
    alt_token = 264

    # Tree attention 请求
    tree_lps = get_logprobs(generate(
        server,
        input_ids=main_ids + [alt_token],
        parent_ids=[-1, 0, 1, 2, 3, 1],
        sampling_params={"max_new_tokens": 0},
        return_logprob=True,
        logprob_start_len=0,
        top_logprobs_num=5,
    ))

    # 对照: 主序列线性请求
    main_lps = get_logprobs(generate(
        server,
        input_ids=main_ids,
        sampling_params={"max_new_tokens": 0},
        return_logprob=True,
        logprob_start_len=0,
    ))

    # 对照: alt 分支线性请求 [0, 1, alt]
    alt_lps = get_logprobs(generate(
        server,
        input_ids=[main_ids[0], main_ids[1], alt_token],
        sampling_params={"max_new_tokens": 0},
        return_logprob=True,
        logprob_start_len=0,
    ))

    ok = True
    # 主序列 logprob 应完全一致
    for i in range(len(main_lps)):
        if not check(f"main[{i}]", tree_lps[i][0], main_lps[i][0]):
            ok = False

    # alt token logprob 应一致
    if not check("alt[5]", tree_lps[5][0], alt_lps[2][0]):
        ok = False

    if ok:
        print(f"    主序列 logprob 完全匹配, alt logprob={tree_lps[5][0]:.4f} 匹配 - PASS")
    return ok


def test_multi_branch(server):
    """多分支树: 2 个不同位置的替代 token"""
    print("\nTest 3: 多分支树 (5 tokens + 2 alts)")
    #
    # 树结构:
    #   0 -> 1 -> 2 -> 3 -> 4
    #             \-> 5 (alt, parent=1)
    #                       \-> 6 (alt, parent=3)
    #
    tree_ids = [9707, 1879, 11, 419, 374, 264, 279]
    tree_pids = [-1, 0, 1, 2, 3, 1, 3]

    tree_lps = get_logprobs(generate(
        server,
        input_ids=tree_ids,
        parent_ids=tree_pids,
        sampling_params={"max_new_tokens": 0},
        return_logprob=True,
        logprob_start_len=0,
    ))

    # 对照: 主序列
    main_lps = get_logprobs(generate(
        server,
        input_ids=[9707, 1879, 11, 419, 374],
        sampling_params={"max_new_tokens": 0},
        return_logprob=True,
        logprob_start_len=0,
    ))

    # 对照: alt1 路径 [0, 1, 264]
    alt1_lps = get_logprobs(generate(
        server,
        input_ids=[9707, 1879, 264],
        sampling_params={"max_new_tokens": 0},
        return_logprob=True,
        logprob_start_len=0,
    ))

    # 对照: alt2 路径 [0, 1, 2, 3, 279]
    alt2_lps = get_logprobs(generate(
        server,
        input_ids=[9707, 1879, 11, 419, 279],
        sampling_params={"max_new_tokens": 0},
        return_logprob=True,
        logprob_start_len=0,
    ))

    ok = True
    for i in range(len(main_lps)):
        if not check(f"main[{i}]", tree_lps[i][0], main_lps[i][0]):
            ok = False

    if not check("alt1[5]", tree_lps[5][0], alt1_lps[2][0]):
        ok = False
    if not check("alt2[6]", tree_lps[6][0], alt2_lps[4][0]):
        ok = False

    if ok:
        print(f"    主序列匹配, alt1={tree_lps[5][0]:.4f}, alt2={tree_lps[6][0]:.4f} 均匹配 - PASS")
    return ok


def test_deep_branch(server):
    """深层分支: 分支上的分支"""
    print("\nTest 4: 深层树 (分支上再分支)")
    #
    # 树结构:
    #   0 -> 1 -> 2 -> 3
    #         \-> 4 -> 5
    #              \-> 6
    #
    tree_ids = [9707, 1879, 11, 419, 264, 279, 374]
    tree_pids = [-1, 0, 1, 2, 1, 4, 4]

    tree_lps = get_logprobs(generate(
        server,
        input_ids=tree_ids,
        parent_ids=tree_pids,
        sampling_params={"max_new_tokens": 0},
        return_logprob=True,
        logprob_start_len=0,
    ))

    # 路径 [0, 1, 4, 5] = [9707, 1879, 264, 279]
    path5_lps = get_logprobs(generate(
        server,
        input_ids=[9707, 1879, 264, 279],
        sampling_params={"max_new_tokens": 0},
        return_logprob=True,
        logprob_start_len=0,
    ))

    # 路径 [0, 1, 4, 6] = [9707, 1879, 264, 374]
    path6_lps = get_logprobs(generate(
        server,
        input_ids=[9707, 1879, 264, 374],
        sampling_params={"max_new_tokens": 0},
        return_logprob=True,
        logprob_start_len=0,
    ))

    ok = True
    # token 4 (264, parent=1): P(264 | logits[1])
    if not check("node4", tree_lps[4][0], path5_lps[2][0]):
        ok = False
    # token 5 (279, parent=4): P(279 | logits[4])
    if not check("node5", tree_lps[5][0], path5_lps[3][0]):
        ok = False
    # token 6 (374, parent=4): P(374 | logits[4])
    if not check("node6", tree_lps[6][0], path6_lps[3][0]):
        ok = False

    if ok:
        print(f"    node4={tree_lps[4][0]:.4f}, node5={tree_lps[5][0]:.4f}, node6={tree_lps[6][0]:.4f} 均匹配 - PASS")
    return ok


def test_longer_sequence(server):
    """较长序列: 用 text 编码后构建树"""
    print("\nTest 5: 较长序列 + 分支")

    # 先获取一段文本的 token ids
    base_result = generate(
        server,
        text="Mathematics is the queen of the sciences and number theory is the queen of mathematics",
        sampling_params={"max_new_tokens": 0},
        return_logprob=True,
        logprob_start_len=0,
        top_logprobs_num=5,
    )
    base_lps = get_logprobs(base_result)
    base_ids = [lp[1] for lp in base_lps]
    n = len(base_ids)
    print(f"    基础序列: {n} tokens")

    if n < 10:
        print("    序列太短，跳过")
        return True

    # 从 top_logprobs 中取几个替代 token
    top_logprobs = base_result["meta_info"]["input_top_logprobs"]
    alt_positions = []
    alt_tokens = []

    for pos in [3, 7, 12]:
        if pos < n and top_logprobs[pos]:
            # 取一个不同于原 token 的候选
            for candidate in top_logprobs[pos]:
                if candidate[1] != base_ids[pos]:
                    alt_positions.append(pos)
                    alt_tokens.append(candidate[1])
                    break

    if not alt_positions:
        print("    找不到替代 token，跳过")
        return True

    # 构建树: 主序列 + 各替代 token
    tree_ids = list(base_ids)
    tree_pids = [-1] + list(range(n - 1))  # 线性链

    for pos, alt_tok in zip(alt_positions, alt_tokens):
        tree_ids.append(alt_tok)
        tree_pids.append(pos - 1)  # parent = 被替代位置的前一个

    tree_lps = get_logprobs(generate(
        server,
        input_ids=tree_ids,
        parent_ids=tree_pids,
        sampling_params={"max_new_tokens": 0},
        return_logprob=True,
        logprob_start_len=0,
    ))

    ok = True
    # 主序列应一致
    for i in range(n):
        if not check(f"main[{i}]", tree_lps[i][0], base_lps[i][0]):
            ok = False

    # 各替代 token 应与独立线性请求一致
    for idx, (pos, alt_tok) in enumerate(zip(alt_positions, alt_tokens)):
        # 线性路径: base_ids[0:pos] + [alt_tok]
        lin_path = list(base_ids[:pos]) + [alt_tok]
        lin_lps = get_logprobs(generate(
            server,
            input_ids=lin_path,
            sampling_params={"max_new_tokens": 0},
            return_logprob=True,
            logprob_start_len=0,
        ))
        tree_idx = n + idx
        if not check(f"alt@{pos}", tree_lps[tree_idx][0], lin_lps[pos][0]):
            ok = False

    if ok:
        print(f"    {n} tokens 主序列 + {len(alt_positions)} 个替代分支全部匹配 - PASS")
    return ok


def main():
    parser = argparse.ArgumentParser(description="Tree Attention 测试")
    parser.add_argument("--server", default="http://localhost:8000", help="SGLang server URL")
    args = parser.parse_args()

    server = args.server.rstrip("/")

    # 检查服务器
    print(f"服务器: {server}")
    try:
        info = requests.get(f"{server}/get_model_info", timeout=5).json()
        print(f"模型: {info['model_path']}\n")
    except Exception as e:
        print(f"无法连接服务器: {e}")
        sys.exit(1)

    tests = [
        test_normal_request,
        test_simple_tree,
        test_multi_branch,
        test_deep_branch,
        test_longer_sequence,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            if test(server):
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"    ERROR: {e}")
            failed += 1

    print(f"\n{'=' * 40}")
    print(f"结果: {passed} passed, {failed} failed")
    if failed > 0:
        sys.exit(1)
    else:
        print("ALL TESTS PASSED!")


if __name__ == "__main__":
    main()
