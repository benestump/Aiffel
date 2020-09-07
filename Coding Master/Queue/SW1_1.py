import sys

sys.stdin = open("./SW1.txt", "r")

T = int(input())

for test_case in range(1, T + 1):
    # ///////////////////////////////////////////////////////////////////////////////////
    ls_num, rotation_num = map(int, input().split(' '))
    input_ls = list(map(int, input().split(' ')))


    print(f'#{test_case} {input_ls[rotation_num % ls_num]}')
