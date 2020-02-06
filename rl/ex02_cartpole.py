import gym
import random

if __name__ == '__main__':
    # 게임 environment 생성
    env = gym.make('CartPole-v1')

    # 게임 환경 초기화
    obs = env.reset()
    # 초기화 화면 출력
    env.render()
    print(obs)

    max_steps = 1000  # 최대 반복 회수
    # for문 반복할 때마다 action 값이 0 또는 1을 랜덤하게 선택하도록
    # done 값이 True이면 for loop을 종료하도록
    # 몇 번 step만에 게임이 종료됐는지 출력
    for t in range(max_steps):
        action = random.randint(0, 1)  # 게임 액션 설정
        obs, reward, done, info = env.step(action)  # 게임 진행
        env.render()  # 게임 환경 화면 출력
        print(obs)
        print(f'reward: {reward}, done: {done}, info: {info}')
        if done:
            print('finished after step', t + 1)
            break

    env.close()  # 게임 환경 종료
