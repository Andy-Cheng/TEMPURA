import React from 'react';
import { Typography } from 'antd';
import Video from '../../components/Video';
import Image from '../../components/Image';
import Paragraph from '../../components/Paragraph';
import Parent from './Parent';
import impossibleStaircaseBanner from '../../images/impossible-staircase_banner.jpg';
import {
  Anchor
} from './Parent.style';

const HapticGoRound = () => (
  <Parent
    bannerSrc={impossibleStaircaseBanner}
    title="Impossible Staircase: Vertically Real Walking in an Infinite Virtual Tower"
    publication="IEEE VR '21"
    link="https://ieeexplore.ieee.org/document/9417791"
    authors="Jen-Hao Cheng; Yi Chen; Ting-Yi Chang; Hsu-En Lin; Po-Yao Cosmos Wang; Lung-Pan Cheng"
    time="Jun. 2020 - Nov. 2020"
    location="Taipei, Taiwan"
    category="Research"
  >
    <Typography>
      <Video url="https://youtu.be/7hb0TJvDrkY?si=pa-xbDhRa9L6UVLI" />
      <Paragraph>
            We present Impossible Staircase, a real-walking virtual reality system that allows users to climb an infinite virtual tower. Our set-up
      consists of an one-level scaffold and a lifter. A user climbs up the
      scaffold by real walking on a stairway while wearing a head-mounted
      display, and gets reset to the ground level by a lifter imperceptibly.
      By repeating this process, the user perceives an illusion of climbing
      an infinite number of levels. Our system achieves the illusion by
      (1) controlling the movement of the lifter to generate reverse and
      imperceptible motion, (2) guiding the user through the scaffold with
      delay mechanisms to reset the lifter in time, and (3) procedural generating overlapping structures to enlarge perceived height of each
      level. We built a working system and demonstrated it with a 15-min
      experience. With the working system, we conducted user studies to
      gain deeper insights into vertical motion simulation and vertical real
      walking in virtual reality.
      </Paragraph>
    </Typography>

    Source code is available at: <Anchor href="https://github.com/Andy-Cheng/Impossible-Staircase">https://github.com/Andy-Cheng/Impossible-Staircase</Anchor>.
  </Parent>
);

export default HapticGoRound;
