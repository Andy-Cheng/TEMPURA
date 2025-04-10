import React from 'react';
import styled from 'styled-components';

const Container = styled.div`
  background-color: black;
  height: calc(100vh - 134px); // 64px + 70px
  display: flex;
  justify-content: center;
  align-items: center;
`;

const Text = styled.div`
  color: white;
  font-size: 60px;
  font-weight: 600;
`;

const ComingSoon = () => (
  <Container>
    <Text>Coming Soon</Text>
  </Container>
);

export default ComingSoon;
