import styled from 'styled-components';

export const Container = styled.div`
  background-color: white;
  display: flex;
  flex-direction: column;
`;

export const ContentOuter = styled.div`
  max-width: 1440px;
  padding: 40px 120px;
  margin: 0 auto;
  box-sizing: border-box;
  @media (max-width: 768px) {
    padding: 40px 20px;
  }
`;

export const ContentInner = styled.div`
  margin: auto;
`;

export const Banner = styled.div`
  width: 100%;
  height: 400px;
  background-repeat: no-repeat;
  background-position: center;
  background-size: cover;
  @media (max-width: 768px) {
    height: 300px;
  }
  @media (max-width: 480px) {
    height: 250px;
  }
`;

export const Title = styled.div`
  color: black;
  font-size: 32px;
  font-weight: 600;
  text-align: center;
`;

export const BadgeContainer = styled.div`
  display: flex;
  flex-direction: column;
  margin: 32px auto;
  width: 60%;
  @media (max-width: 768px) {
    width: 70%;
  }
  @media (max-width: 480px) {
    width: 100%;
  }
`;

export const Children = styled.div`
  margin-top: 32px;
  color: black;
`;

export const Anchor = styled.a.attrs({
  target: '_blank',
  rel: 'noopener noreferrer'
})`
  color: darkblue;
`;
