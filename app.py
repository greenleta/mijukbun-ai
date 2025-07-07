import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("📉 미적분 기반 경사하강법 시뮬레이터")

st.markdown("""
이 시뮬레이터는 함수의 **기울기(미분)**를 이용하여,  
함수의 최소값을 찾아가는 **경사하강법(Gradient Descent)** 과정을 시각화합니다.
""")

# 사용자 입력 받기
lr = st.slider("🔧 학습률 (Learning Rate)", 0.001, 1.0, 0.1)
epochs = st.slider("🔁 반복 횟수", 1, 100, 20)
initial_x = st.slider("📍 시작 지점 x", -10.0, 10.0, 5.0)

# 선택 가능한 함수
func_choice = st.selectbox("📐 최적화할 함수 선택", ["x²", "x⁴ - 3x²", "sin(x) + x²"])

# 함수 정의
def f(x):
    if func_choice == "x²":
        return x**2
    elif func_choice == "x⁴ - 3x²":
        return x**4 - 3*x**2
    elif func_choice == "sin(x) + x²":
        return np.sin(x) + x**2

def df(x):
    if func_choice == "x²":
        return 2*x
    elif func_choice == "x⁴ - 3x²":
        return 4*x**3 - 6*x
    elif func_choice == "sin(x) + x²":
        return np.cos(x) + 2*x

# 경사하강법 수행
x_vals = [initial_x]
y_vals = [f(initial_x)]
x = initial_x

for _ in range(epochs):
    grad = df(x)
    x -= lr * grad
    x_vals.append(x)
    y_vals.append(f(x))

# 결과 출력
st.subheader("📈 경사하강법 최적화 경로")

fig, ax = plt.subplots()
X = np.linspace(-10, 10, 500)
Y = f(X)

ax.plot(X, Y, label=f"f(x) = {func_choice}")
ax.plot(x_vals, y_vals, 'ro--', label="경사하강법 경로")
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.set_title("경사하강법을 통한 최소값 탐색")
ax.legend()
st.pyplot(fig)

# 수학적 결과 요약
st.markdown(f"🧮 **최종 x값**: `{x_vals[-1]:.4f}`")
st.markdown(f"🔽 **최종 f(x)**: `{y_vals[-1]:.4f}`")
