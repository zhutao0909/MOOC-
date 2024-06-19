import streamlit as st
import pandas as pd
import subprocess

from Spider import get_MOOC
from draw import set_chinese_font, word_frequency_analysis, word_cloud, score_proportion, \
    plot_comment_frequency_by_month, TSNE_show, word_fre_draw
from utils import chinese_word_cut, vectorization_comment, KmeansAlgorithm


# 读取csv文件中的评论数据
def read_comments_from_csv(file_name):
    df = pd.read_csv(file_name)
    return df['comment'].tolist()


def run_crawler():
    subprocess.run(["python", "F:\python练习代码\pythonProject8\MOOC\Spider.py"])

def print_terms_perCluster(true_k,model, data):
    # 输出每个簇的元素
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = model.labels_
    list = []
    for i in range(true_k):
        for ind in order_centroids[i, :2]: # 每簇只选前2个
            list.append(data['split'][ind])
        st.write(f'簇{i}: {", ".join(list)}')
        list = []



# 主函数
def main():
    # 全局变量设置①---页面参数设置
    if 'url' not in st.session_state:
        st.session_state.url = 'https://www.icourse163.org/course/BIT-268001?outVendor=zw_mooc_pcsybzkcph_'
    if 'num_pages' not in st.session_state:
        st.session_state.num_pages = 20
    if 'browser' not in st.session_state:
        st.session_state.browser = 'chrome'
    if 'min_df' not in st.session_state:
        st.session_state.min_df = 5
    if 'true_k' not in st.session_state:
        st.session_state.true_k = 5
    if 'max_iter' not in st.session_state:
        st.session_state.max_iter = 100
    if 'n_init' not in st.session_state:
        st.session_state.n_init = 1

    # 全局变量设置②---系统过程参数设置
    st.session_state.filenamePath = ""  # 操作文件地址
    st.session_state.data = None  # 文件对象
    st.session_state.model = None  # 模型对象
    st.session_state.vectorizer = None  # 文件对象


    st.title('MOOC在线课程评论分析')
    page = st.sidebar.selectbox('选择页面',
                                ['配置', '爬取到的评论数据展示', '评论数据分析结果展示'])

    if page == '配置':
        st.header('爬虫模块设置')
        st.session_state.url = st.text_input('输入待爬取的路径', st.session_state.url)
        st.session_state.num_pages = st.number_input('要爬取页面的数量', min_value=1, value=st.session_state.num_pages)
        st.session_state.browser = st.selectbox('使用的浏览器驱动', ['chrome', 'firefox', 'edge'],
                                                index=['chrome', 'firefox', 'edge'].index(st.session_state.browser))

        if st.button('确认'):
            st.session_state.filenamePath = get_MOOC(st.session_state.browser, st.session_state.url, st.session_state.num_pages)

        st.header('KMeans算法模块设置')
        st.session_state.min_df = st.number_input('min_df数据大小', min_value=1, value=st.session_state.min_df)
        st.session_state.true_k = st.number_input('true_k数据大小', min_value=1, value=st.session_state.true_k)
        st.session_state.max_iter = st.number_input('max_iter数据大小', min_value=1, value=st.session_state.max_iter)
        st.session_state.n_init = st.number_input('n_init数据大小', min_value=1, value=st.session_state.n_init)

    elif page == '爬取到的评论数据展示':
        st.header('爬取到的评论数据展示')
        file_name = st.text_input('输入csv文件地址',st.session_state.filenamePath)
        if st.button('读取数据'):
            df = pd.read_csv(file_name) # 显示这个表格数据
            st.write(df)



    elif page == '评论数据分析结果展示':
        st.header('评论数据分析结果展示')
        commentPath = st.text_input('输入csv文件地址',st.session_state.filenamePath)
        if st.button('分析数据'):

            # 在页面上添加一个分界线
            st.markdown('<hr>', unsafe_allow_html=True)
            st.subheader("评论数据进行预处理之后，对得到的统计分析结果进行可视化展示：")  # 添加一个二级标题

            # -----对评论数据进行预处理------------
            st.session_state.data = pd.read_csv(commentPath, encoding='utf-8')
            st.session_state.data['split'] = st.session_state.data['Comment'].apply(chinese_word_cut)
            set_chinese_font()  # 设置中文展示字体

            # -----数据可视化对其进行统计分析------------
            col1, col2 = st.columns(2)
            with col1:
                st.write('词汇频率分析')
                st.write(word_fre_draw((word_frequency_analysis(st.session_state.data)), 'All'))
                words = word_frequency_analysis(st.session_state.data)  # 词汇频率分析
                st.write('词云')
                st.write(word_cloud(words)) # 词云制作

            with col2:
                st.write('评分占比情况')
                st.write(score_proportion(st.session_state.data)) # 爬取课程评论评分的占比情况分析(饼状图)

                st.write('发布时间统计')
                st.write(plot_comment_frequency_by_month(st.session_state.data)) # 发布时间统计情况


            # -----评论向量化------------
            [st.session_state.vectorizer,tfidf_weight] = vectorization_comment(st.session_state.min_df,st.session_state.data)

            # -----使用K-means聚类算法对评论数据进行情感分类-----------
            st.session_state.model = KmeansAlgorithm(st.session_state.data, tfidf_weight,st.session_state.max_iter,st.session_state.n_init)



            # 在页面上添加一个分界线
            st.markdown('<hr>', unsafe_allow_html=True)
            st.subheader("使用聚类算法，对评论数据进行情感分类分析：") # 添加一个二级标题

            col3, col4 = st.columns(2)  # 创建一个包含两列的布局

            with col3:
                # -----使用聚类分析之后，数据可视化-----------
                # 使用T-SNE算法，对权重进行降维，准确度比PCA算法高，但是耗时长
                st.write('T-SNE降维')
                st.write(TSNE_show(tfidf_weight, st.session_state.model))

            with col4:
                # -----输出每个簇的元素-----------
                st.write('分簇词条')
                print_terms_perCluster(st.session_state.true_k,st.session_state.model, st.session_state.data)

if __name__ == '__main__':
    main()
