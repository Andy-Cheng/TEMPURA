import React from 'react';
import { BrowserRouter as Router, Switch, Route, Link } from 'react-router-dom';
import { Layout, Menu } from 'antd';
import Works from './Pages/Works';
import Project from './Pages/Project';
import About from './Pages/About';
import NotFound from './Pages/NotFound';
import ComingSoon from './Pages/ComingSoon';
import './App.less';
import { Header, Title } from './App.style';

const { Content, Footer } = Layout;
const { Item } = Menu;

const App = () => (
  <Router>
    <Layout style={{ backgroundColor: '#000', height: '100%' }}>
      <Header
        style={{
          position: 'fixed',
          zIndex: 1,
          width: '100%',
          backgroundColor: '#fff',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between'
        }}
      >
        <Link to="/">
          <Title level={4} style={{ marginBottom: 0 }}>
            J.C.
          </Title>
        </Link>
        <Menu
          mode="horizontal"
          style={{
            display: 'flex',
            flexDirection: 'row',
            justifyContent: 'flex-end'
          }}
          defaultSelectedKeys={['1']}
        >
          <Item key="1">
            <Link to="/">About</Link>
          </Item>
          <Item key="2">
            <Link to="/works">Works</Link>
          </Item>
          <Item key="3">
            <Link to="/notes">My Notes</Link>
          </Item>
        </Menu>
      </Header>
      <Content
        style={{
          marginTop: 64,
          color: '#fff'
        }}
      >
        <Switch>
          <Route path="/notes">
            <ComingSoon />
          </Route>
          <Route path="/project">
            <Project />
          </Route>
          <Route path="/works">
            <Works />
          </Route>
          <Route path="/">
            <About />
          </Route>
          <Route path="*">
            <NotFound />
          </Route>
        </Switch>
      </Content>
      <Footer style={{ textAlign: 'center' }}>
        Copyright© Jen-Hao Cheng, 2025
        <br />
        
        <a href="https://github.com/Andy-Cheng/Andy-Personal-Website" style={{ color: 'darkblue' }}>
          Website Source Code
        </a> 
      </Footer>
    </Layout>
  </Router>
);

export default App;
