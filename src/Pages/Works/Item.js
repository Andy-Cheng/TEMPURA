import React from 'react';
import { Link } from 'react-router-dom';
import styled from 'styled-components';

const Mask = styled.div`
  width: 100%;
  height: 100%;
  background-color: rgba(0, 0, 0, 0.8);
  display: flex;
  flex-direction: column;
  padding: 32px 24px;
  box-sizing: border-box;
`;

const Container = styled.div`
  width: 600px;
  height: 450px;
  -webkit-box-flex: 0;
  -ms-flex: 0 0 33.333333%;
  flex: 0 0 33.333333%;
  max-width: 33.333333%;
  background-repeat: no-repeat;
  background-position: center;
  background-size: cover;
  & > .link > ${Mask} {
    display: none;
  }
  &:hover {
    cursor: pointer;
    > .link > ${Mask} {
      display: flex;
    }
  }
  @media screen and (max-width: 1200px) {
    width: 400px;
    height: 360px;
    -webkit-box-flex: 0;
    -ms-flex: 0 0 50%;
    flex: 0 0 50%;
    max-width: 50%;
  }
  @media screen and (max-width: 768px) {
    -webkit-box-flex: 0;
    -ms-flex: 0 0 100%;
    flex: 0 0 100%;
    max-width: 100%;
  }
  @media screen and (max-width: 480px) {
    width: 480px;
  }
`;

const Title = styled.h1`
  font-size: 32px;
  color: white;
`;

const Description = styled.p`
  font-size: 16px;
  color: white;
`;

const Item = ({ title, description, mediaSrc, path }) => {
  return (
    <Container style={{ backgroundImage: `url('${mediaSrc}')` }}>
      <Link
        to={{
          pathname: `/project`,
          hash: path
        }}
        className="link"
      >
        <Mask>
          <Title>{title}</Title>
          <Description>{description}</Description>
        </Mask>
      </Link>
    </Container>
  );
};

export default Item;
