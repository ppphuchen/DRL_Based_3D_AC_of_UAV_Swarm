import argparse
import numpy as np
def arglists():
    parser = argparse.ArgumentParser()
    #LUAV的具有默认值的参数
    parser.add_argument("--patch_fuav_num", type=int, default=4, help="子群中FUAV的数量", )
    parser.add_argument("--can_comm", type=float, default=100, help="两个LUAV之间通信的功率阈值")
    parser.add_argument("--patch_num", type=int, default=10, help="子群的默认数量")
    parser.add_argument("--L_power", type=float, default=10, help="LUAV与其他UAV的传输功率")
    parser.add_argument("--L_UAV_num", type=int, default=10, help="环境中的LUAV的默认数量")
    parser.add_argument("--L_comm_radius", type=int, default=250, help="LUAV的通信范围")
    parser.add_argument("--gain_r", type=float, default=1.0, help="LUAV的接收增益，暂定为1")
    parser.add_argument("--gain_t", type=float, default=1.0, help="LUAV的发送增益，暂定为1")
    parser.add_argument("--re_cover_rate", type=float, default=0.05, help="LUAV的重复覆盖率")
    parser.add_argument("--average_collect_time", type=int, default=1, help="子群飞行每米的覆盖时间")
    #FUAV具有的参数
    parser.add_argument("--F_power", type=float, default=10, help="FUAV对LUAV的默认传输功率")
    parser.add_argument("--F_comm_radius", type=int, default=1, help="FUAV的通信范围")
    parser.add_argument("--F_line_v", type=int, default=1, help="FUAV的速度")
    #Env具有的参数
    parser.add_argument("--H", type=int, default=5, help="UAV的飞行高度为5米")
    parser.add_argument("--total_UAV_num", type=int, default=50, help="环境中UAV的总数量")
    parser.add_argument("--max_sub_group_cover_time", type=int, default=0, help="子群中的最大覆盖总时间")
    #3D模型的参数
    parser.add_argument("--L_x", type=int, default=1000, help="地图的横轴长度")
    parser.add_argument("--L_y", type=int, default=1000, help="地图的纵轴长度")
    parser.add_argument("--S_x", type=int, default=25, help="地图采样方格的横轴长度")
    parser.add_argument("--S_y", type=int, default=25, help="地图采样方格的纵轴长度")
    return parser.parse_args()
