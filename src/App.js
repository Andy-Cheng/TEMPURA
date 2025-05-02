import React from 'react';
import { Layout } from 'antd';
import './App.less';
import TEMPURA from './Pages/StandalonePages/TEMPURA';

const { Content, Footer } = Layout;

const AppLayout = () => {
  // const [showHeader, setShowHeader] = useState(true);
  // const location = useLocation();


  return (
    <Layout style={{ backgroundColor: '#000', height: '100%' }}>
      <Content
        style={{
          color: '#fff'
        }}
      >
        <TEMPURA />
      </Content>
      <Footer style={{ textAlign: 'center' }}>
        Jen-Hao Cheng, 2025
        <br />
        <a href="https://github.com/Andy-Cheng/Andy-Personal-Website" style={{ color: 'darkblue' }}>
          Website Source Code
        </a>
      </Footer>
    </Layout>
  );
};

const App = () => (
    <AppLayout />
);

export default App;
