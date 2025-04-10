import React from 'react';
import { Typography } from 'antd';
import Video from '../../components/Video';
import Image from '../../components/Image';
import Paragraph from '../../components/Paragraph';
import Header3 from '../../components/Headers';
import Parent from './Parent';
import CenterRadarNetBanner from '../../images/cenrad_banner.jpg';
import {
  Anchor
} from './Parent.style';

const CenterRadarNet = () => (
  <Parent
    bannerSrc={CenterRadarNetBanner}
    title="CenterRadarNet: A Joint Architecture for 3D Object Detection and Re-Identification from 4D Radar Data"
    publication="IEEE ICIP '24"
    link="https://ieeexplore.ieee.org/document/10648077"
    authors="Jen-Hao Cheng; Sheng-Yao Kuan; Hou-I Liu; Hugo Latapie; Gaowen Liu; Jenq-Neng Hwang"
    location="Seattle, WA, USA"
    time="Sep. 2023 - Feb. 2024"
    category="Research"
  >
    <Typography>
      <Video url="https://youtu.be/KxCib_1JkoI?si=C14jJWtkOmmRnpp7" />
      <Paragraph>
            Robust perception is a vital component for ensuring safe autonomous driving. Automotive radar (77 to 81 GHz) offering
      weather-resilient sensing provides a complementary capability to the vision- or LiDAR-based autonomous driving systems. Raw radio-frequency (RF) radar tensors contain rich
      spatiotemporal semantics besides 3D location information.
      Most previous methods take in 3D (Doppler-range-azimuth)
      RF radar tensors, allowing prediction of an object’s location,
      heading angle, and size in bird’s-eye-view (BEV). However,
      they lack the ability to simultaneously infer objects’ size,
      orientation, and identity in the 3D space. To overcome this
      limitation, we propose a joint architecture, called CenterRadarNet, designed to facilitate high-resolution representation learning from 4D (Doppler-range-azimuth-elevation)
      radar data for 3D object detection and re-identification (reID) tasks. Moreover, we build an online tracker utilizing the
      learned appearance embedding for re-ID. CenterRadarNet
      achieves the state-of-the-art result on the K-Radar 3D object
      detection benchmark. In addition, we present the first 3D
      object-tracking result on the K-Radar dataset. CenterRadarNet shows consistent, robust performance in diverse driving
      scenarios, emphasizing its wide applicability.
      </Paragraph>
      
    </Typography>
    Source code is available at:  <Anchor href="https://github.com/Andy-Cheng/CenterRadarNet">https://github.com/Andy-Cheng/CenterRadarNet</Anchor>.
  </Parent>
);

export default CenterRadarNet;
