"""ex11.py에서 저장한 pickle 파일을 읽어서 파라미터를 화면에 출력"""
import pickle

if __name__ == '__main__':
    with open('mnist_param.pkl', 'rb') as f:
        params = pickle.load(f)
    for key in params:
        print(key, params[key].shape)
        print(params[key][:5])
