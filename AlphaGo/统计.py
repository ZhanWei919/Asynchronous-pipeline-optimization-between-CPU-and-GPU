import re

log_file = "game_log.txt"
game_stats = []  # 存储格式改为元组 (总耗时, 传输次数, 搜索次数)

current_game_time = 0
current_game_pcie = 0
search_count = 0  # 新增搜索次数计数器

with open(log_file, 'r', encoding='utf-8') as file:
    for line in file:
        # 提取搜索耗时
        if match := re.search(r"搜索耗时:\s*(\d+\.\d+) 秒", line):
            current_game_time += float(match.group(1))
            search_count += 1  # 每次搜索计数

        # 提取PCIe传输次数
        if match := re.search(r"PCIe 传输次数:\s*(\d+)", line):
            current_game_pcie += int(match.group(1))

        # 游戏结束标记
        if "Game -" in line:
            game_stats.append((
                current_game_time,
                current_game_pcie,
                search_count
            ))
            # 重置计数器
            current_game_time = 0
            current_game_pcie = 0
            search_count = 0

# 输出结果
for i, (total_time, pcie, searches) in enumerate(game_stats, start=1):
    # 计算三种平均值
    avg_per_search = total_time / searches if searches > 0 else 0
    avg_per_pcie = total_time / pcie if pcie > 0 else 0
    pcie_per_search = pcie / searches if searches > 0 else 0

    print(f"\nGame {i}:")
    print(f"总搜索耗时: {total_time:.6f} 秒")
    print(f"总PCIe传输: {pcie} 次")
    print(f"搜索次数: {searches} 次")
    print(f"平均单次搜索耗时: {avg_per_search:.6f} 秒")
    print(f"平均每次PCIe耗时: {avg_per_pcie:.6f} 秒")
    print(f"每次搜索PCIe次数: {pcie_per_search:.2f} 次")

# 保存到文件
with open("game_summary.txt", 'w', encoding='utf-8') as f:
    f.write("游戏统计摘要\n")
    f.write("=" * 30 + "\n")

    for i, (total_time, pcie, searches) in enumerate(game_stats, start=1):
        avg = total_time / searches if searches > 0 else 0
        f.write(
            f"Game {i}:\n"
            f"总耗时: {total_time:.4f}s | "
            f"PCIe传输: {pcie}次 | "
            f"搜索次数: {searches}次\n"
            f"单次搜索平均: {avg:.6f}s\n"
            f"{'-' * 30}\n"
        )