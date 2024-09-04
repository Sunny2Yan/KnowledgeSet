# -*- coding: utf-8 -*-
import streamlit as st


st.markdown('Streamlit Demo')  # 初始化markdown
st.title('这是一个标题')
st.header('这是一级标题')
st.subheader('这是二级标题')
st.text('这是一个文本')
code1 = '''pip install streamlit'''
st.code(code1, language='bash')  # 代码

code2 = '''import streamlit as st
st.markdown('Streamlit Demo')'''
st.code(code2, language='python')

st.latex("\frac{1}{2}")  # latex 公式
st.caption("这是一个小体文本")  # 小字体文本