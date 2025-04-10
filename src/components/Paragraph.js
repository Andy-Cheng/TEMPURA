import styled from 'styled-components';
import { Typography } from 'antd';

const { Paragraph: AntdParagraph } = Typography;

const Paragraph = styled(AntdParagraph)`
  width: 100%;
  margin-top: 32px;
  font-size: 16px;
`;

export default Paragraph;
