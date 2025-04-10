import React from 'react';
import { Typography } from 'antd';
import Video from '../../components/Video';
import Image from '../../components/Image';
import Paragraph from '../../components/Paragraph';
import Parent from './Parent';
import hapticGoRoundImg1 from '../../images/hapticGoRoundImg1.jpg';
import hapticGoRoundImg2 from '../../images/hapticGoRoundImg2.jpg';
import hapticGoRoundImg3 from '../../images/hapticGoRoundImg3.jpg';
import hapticGoRoundImg4 from '../../images/hapticGoRoundImg4.jpg';
import hapticGoRoundImg5 from '../../images/hapticGoRoundImg5.jpg';

const HapticGoRound = () => (
  <Parent
    bannerSrc={hapticGoRoundImg1}
    title="Haptic-go-round"
    publication="ACM CHI '20"
    link="https://doi.org/10.1145/3313831.3376476"
    authors="Hsin-Yu Huang, Chih-Wei Ning, Po-Yao Wang, Jen-Hao Cheng, and Lung-Pan Cheng"
    time="Aug. 2019 - Jan. 2020"
    location="Taipei, Taiwan"
    category="Research"
  >
    <Typography>
      <Video url="https://youtu.be/55cr0c-cyRA" />
      <Image src={hapticGoRoundImg2} />
      <Paragraph>
        Haptic-go-round (HGR) is a surrounding platform (a) that allows
        deploying props and devices (b) to provide haptic feedback from any
        direction for virtual reality experiences. The key component of HGR is a
        motorized turntable that rotates the correct haptic device to the right
        direction at the right time to match what users are about to touch (c,
        d).
      </Paragraph>
      <Image src={hapticGoRoundImg3} />
      <Paragraph>
        It also provides force feedback. The picture above shows a user is
        dragged by a virtual marlin around the ship as the turntable rotates.
      </Paragraph>
      <Image src={hapticGoRoundImg4} />
      <Paragraph>
        We implemented a working platform including plug-and-play prop
        cartridges and a software interface (as shown in the picture below) that allows experienced designers to
        agilely add their haptic components and use the platform for their
        applications.
      </Paragraph>
      <Image src={hapticGoRoundImg5} />
    </Typography>
  </Parent>
);

export default HapticGoRound;
