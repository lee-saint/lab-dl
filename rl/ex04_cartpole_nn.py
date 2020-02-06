"""
CartPole 게임에 Neural Network 적용
"""
import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras


def render_policy_net(model, max_steps=200):
    """ 신경망 모델과 최대 step 회수가 주어지면 게임 화면을 출력하는 함수"""
    env = gym.make('CartPole-v1')  # 게임 환경 생성
    obs = env.reset()  # 게임 환경 초기화
    env.render()  # 초기 화면 렌더링
    for step in range(max_steps):  # 최대 스텝 회수만큼 반복
        p = model.predict(obs.reshape(1, -1))
        # 예측값 p를 이용해서 action을 결정(0: 왼쪽, 1: 오른쪽)
        action = int(np.random.random() > p)
        # action을 게임 환경에 적용(step) -> 다음 스텝(환경)으로 변화
        obs, reward, done, info = env.step(action)
        env.render()  # 바뀐 환경을 렌더링
        if done:  # 게임이 종료되면
            print(f'--- Finished after step {step + 1} ---')
            break
    if not done:  # max_step 번 반복하는 동안 게임이 종료되지 않았을 때
        print('Still Alive!!!')
    # 게임 환경 종료
    env.close()


if __name__ == '__main__':
    # 신경망 생성
    model = keras.Sequential()  # keras.models.Sequential()
    # fully-connected 은닉층(hidden layer)을 추가
    model.add(keras.layers.Dense(4, activation='elu', input_shape=(4,)))
    # fully-connected 출력층을 추가
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    # 신경망 요약 정보
    model.summary()

    # ReLU(Rectified Linear Unit)
    # f(x) = x if x > 0,
    #      = 0 otherwise
    # ELU(Exponential Linear Unit)
    # f(x) = x, if x > 0,
    #      = alpha * (exp(x) - 1), otherwise
    # 0 <= alpha < 1

    # CartPole 게임 환경(environment) 생성
    env = gym.make('CartPole-v1')
    obs = env.reset()  # 환경 초기화
    print(obs)

    # observation을 신경망 모델에 forward 시켜 예측값을 알아냄
    p = model.predict(obs.reshape(1, -1))
    print('p=', p)

    action = int(np.random.random() > p)
    print('action =', action)
    # p 값이 1에 가까울수록 np.random.random() > p는 False가 될 확률이 높아짐 -> action이 0이 될 확률이 높아짐
    # -> action이 0이 될 확률이 높아짐
    # -> action 0: 카트를 왼쪽으로 가속도 주겠다는 의미
    # p 값이 0에 가까울수록 np.random.random() > p는 True가 될 확률이 높아짐
    # -> action은 1이 될 확률이 높아짐
    # -> action 1: 카트를 오른쪽으로 가속도 주겠다는 의미

    env.close()  # 게임 환경 종료

    # render_policy_net() 함수에 신경망 모델을 전달해서 게임 실행
    render_policy_net(model)

    # 신경망의 학습 가능 파라미터(weight, bias)를 mini-batch를 사용해 학습시킴
    n_envs = 50  # 학습에 사용할 게임 환경(environments)
    n_iterations = 1000  # 학습 회수

    # 게임 환경 50개 생성
    environments = [gym.make('CartPole-v1') for _ in range(n_envs)]
    # 게임 환경 50개 초기화
    observations = [env.reset() for env in environments]
    # gradient를 업데이트하는 방법 선택
    optimizer = keras.optimizers.RMSprop()
    # 손실 함수(loss function) 선택
    loss_fn = keras.losses.binary_crossentropy

    # 학습
    for iteration in range(n_iterations):
        # Deep Learning에서 Loss를 계산하기 위해서는 target(정답)과 prediction(예측값)이 있어야 함
        # Reinforcement Learning에는 target이 없음!
        # loss를 정의해야 함
        # target을 정읭하기 위한 정책(policy):
        #   angle > 0이면 target = 0, angle < 0 이면 target = 1
        target_probs = np.array([
            ([0.] if obs[2] > 0 else [1.]) for obs in observations
        ])
        with tf.GradientTape() as tape:
            left_probs = model(np.array(observations))
            loss = tf.reduce_mean(loss_fn(target_probs, left_probs))
        print(f'Iteration #{iteration}: Loss={loss.numpy()}')
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        actions = (np.random.rand(n_envs, 1) > left_probs.numpy()).astype(np.int32)
        for idx, env in enumerate(environments):
            obs, reward, done, info = env.step(actions[idx][0])
            observations[idx] = obs if not done else env.reset()

    # 생성된 모든 게임 환경 종료
    for env in environments:
        env.close()

    # 학습이 끝난 모델을 이용해서 게임 실행
    render_policy_net(model)
