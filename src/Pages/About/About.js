import React from 'react';
import { Typography, Avatar, Timeline } from 'antd';
import Paragraph from '../../components/Paragraph';
import { ClockCircleOutlined, HomeOutlined, FileOutlined, LikeOutlined } from '@ant-design/icons';
import { Link } from 'react-router-dom';
import { GithubIcon, GoogleScholarIcon, YouTubeIcon, UWHuskyIcon } from '../../components/Icons';
import {
    Container,
    ContentOuter,
    ContentInner,
    Banner,
    Title,
    BadgeContainer,
    Children,
    Anchor
} from '../Project/Parent.style';
import AndyImg from '../../images/andy.jpg';
import * as PROJECTS from '../Project/projects.const';
import cv from '../../documents/Andy_CV_042025.pdf';
import { TimelineItemShort, TimelineItemMedium, TimelineItemLong1, TimelineItemLong2 } from '../../components/Timelines';
const About = () => (
    <Container style={{ marginTop: 64 }}>
        <ContentOuter>
            <ContentInner>
                <Avatar style={{ marginLeft: "50%", transform: "translateX(-50%)" }} size={150} src={AndyImg} />
                <Title>About Me</Title>
                <Typography.Title level={5} style={{ color: "gray", marginBottom: "10px", marginTop: "40px", textAlign: "center" }}>
                    AI Researcher / VR Developer / Web Developer
                </Typography.Title>
                <Typography.Title level={5} style={{ color: "gray", marginBottom: "10px", marginTop: "20px", textAlign: "center" }}>
                    <Anchor href="https://github.com/Andy-Cheng"><GithubIcon style={{ fontSize: "40px", marginRight: "10px" }} /></Anchor>  <Anchor href="https://scholar.google.com/citations?user=UrH4PK4AAAAJ"><GoogleScholarIcon style={{ fontSize: "40px", marginRight: "10px" }} /></Anchor>  <Anchor href="https://www.youtube.com/@jen-haocheng8594"><YouTubeIcon style={{ fontSize: "40px", marginRight: "10px" }} /></Anchor>
                </Typography.Title>
                <Typography.Title level={5} style={{ color: "gray", textAlign: "center", textDecoration: "underline"}}>
                <Anchor href={cv} target="_blank" rel="noopener noreferrer" style={{ color: "gray", fontWeight: "lighter" }}>
                    My Resume
                </Anchor>
                </Typography.Title>

                <Paragraph>
                    Hi, I’m Jen-Hao Cheng, but I go by Andy. I’m currently a Ph.D. student at the University of Washington, advised by <Anchor href="https://people.ece.uw.edu/hwang/" style={{ color: "darkblue" }}>Dr. Jenq-Neng Hwang</Anchor> in the <Anchor href="https://ipl-uw.github.io/" style={{ color: "purple" }}>Information Processing Lab (IPL)</Anchor>. My research interests lie in understanding how humans perceive the world and enabling computers to perceive and interpret it in similar ways. This passion drives my work at the intersection of human-computer interaction, computer vision, and multi-modal language models.
                </Paragraph>

                <Paragraph>
                    I spent three wonderful years as an undergraduate research assistant at National Taiwan University (NTU). I was incredibly lucky to be mentored by my favorite researcher in the world, <Anchor href="https://www.lungpancheng.tw/" style={{ color: "darkblue" }}>Dr. Lung-Pan Cheng</Anchor>. Working with him inspired me to pursue a Ph.D., as he showed me how to appreciate the beauty of research. From him, I learned how to think critically, turn bold and unconventional ideas into reality, and grow into a better leader.
                    Most importantly, beyond technical skills, he taught me how research can impact society and improve people’s lives.
                    I’m also fortunate to have had <Anchor href="https://www.ee.ntu.edu.tw/profile1.php?id=66" style={{ color: "darkblue" }}>Dr. Wanjiun Liao</Anchor> as my undergraduate advisor. She emphasized the value of strong fundamentals and encouraged me to think creatively and embrace innovative thinking.
                </Paragraph>


                <Paragraph>
                    <Typography.Title level={4} italic style={{ marginBottom: "40px", marginTop: "40px", textAlign: "center", fontWeight: "lighter" }}>
                    Life is a journey, and I feel fortunate to have met many incredible people along the way who have helped me grow.
                    </Typography.Title>
                    <Timeline mode="alternate" reverse={true}>
                        <TimelineItemShort dot={<HomeOutlined style={{ fontSize: '14px' }} />}>
                            <Typography.Text style={{ color: "orange" }}>
                                Attended NTU, 2015-09
                            </Typography.Text>
                        </TimelineItemShort>

                        <TimelineItemMedium dot={<ClockCircleOutlined style={{ fontSize: '14px' }} />} >
                            <Typography.Text >
                                Joined Internet Research Lab (IRL) at NTU EE working with Dr. Wanjiun Liao, 2018-07
                            </Typography.Text>
                        </TimelineItemMedium>
                        <TimelineItemLong1>
                            <Typography.Text >
                                Implemented a 5G and Multi-Access Edge Computing (MEC) enabled video streaming service for mobile
                                VR devices implemented with Unity3D. <Anchor href="https://www.youtube.com/watch?v=ydrUE56VBlc" style={{ color: "darkblue" }}>Edge VR Demo (IEEE ICCE-TW)</Anchor>
                            </Typography.Text>
                        </TimelineItemLong1>

                        <TimelineItemShort dot={<FileOutlined style={{ fontSize: '14px' }} />}>
                            <Typography.Text style={{ color: "orange" }}>
                                Bachelor Degree at EE, NTU , 2019-06
                            </Typography.Text>
                        </TimelineItemShort>

                        <TimelineItemShort dot={<ClockCircleOutlined style={{ fontSize: '14px' }} />} >
                            <Typography.Text >
                                Joined HCI Lab at NTU CSIE working with Dr. Lung-Pan Cheng, 2019-07
                            </Typography.Text>
                        </TimelineItemShort>

                        <TimelineItemLong2>
                            <Typography.Text >
                                Worked on a novel surrounding platform providing hpatics to make VR applications more fun. Hsin-Yu is an amazing team leader. Cosmos and Chih-Wei are great teammates! I enjoyed the time we were chasing the CHI deadline.
                                <Link
                                    to={{
                                        pathname: `/project`,
                                        hash: PROJECTS.HAPTIC_GO_ROUND
                                    }}
                                    className="link"
                                    style={{ color: "darkblue", marginLeft: "2px" }}
                                >
                                    Haptic-go-round (ACM CHI)
                                </Link>
                            </Typography.Text>
                        </TimelineItemLong2>


                        <TimelineItemShort dot={<ClockCircleOutlined style={{ fontSize: '14px' }} />} >
                            <Typography.Text >
                                Left NTU IRL, 2020-03
                            </Typography.Text>
                        </TimelineItemShort>


                        <TimelineItemLong1>
                            <Typography.Text >
                                Led a team to work on a novel VR system to make vertical locomotion more immersive and fun. Published the paper as the first author for the first time.                             <Link
                                    to={{
                                        pathname: `/project`,
                                        hash: PROJECTS.IMPOSSIBLE_STAIRCASE
                                    }}
                                    className="link"
                                    style={{ color: "darkblue", marginLeft: "2px" }}
                                >
                                    Impossible Staircase (IEEE VR)
                                </Link>
                            </Typography.Text>
                        </TimelineItemLong1>

                        <TimelineItemMedium>
                            <Typography.Text >
                                Implemented illusion games. Designed and implemented the illusion database for the research project led by <Anchor href="https://poyaocosmoswang.github.io/" style={{ color: "darkblue" }}>Cosmos</Anchor>.

                                <Anchor href="https://poyaocosmoswang.github.io/works/game_illusionization.html" style={{ color: "darkblue", marginLeft: "2px" }}>Game Illusionization (ACM UIST)</Anchor>
                            </Typography.Text>
                        </TimelineItemMedium>

                        <TimelineItemShort dot={<ClockCircleOutlined style={{ fontSize: '14px' }} />} >
                            <Typography.Text >
                                Left NTU HCI Lab, 2021-03
                            </Typography.Text>
                        </TimelineItemShort>

                        <TimelineItemShort dot={<HomeOutlined style={{ fontSize: '14px' }} />}>
                            <Typography.Text style={{ color: "purple" }}>
                                Attended UW, 2021-09
                            </Typography.Text>
                        </TimelineItemShort>

                        <TimelineItemLong2 dot={<ClockCircleOutlined style={{ fontSize: '14px' }} />} >
                            <Typography.Text >
                                Joined IPL at UW working with <Anchor href="https://scholar.google.com/citations?hl=en&user=QO9Lg_YAAAAJ" style={{ color: "darkblue" }}>Dr. Yizhou Wang</Anchor> and Dr. Jenq-Neng Hwang on autonomous driving, 2022-01
                                <br />
                                Yizhou is a great mentor, and I learned a lot from him about research and life.
                                I was also fortunate to have <Anchor href="https://haotian-zhang.github.io/" style={{ color: "darkblue" }}>Haotian Zhang</Anchor> as my mentor. He taught me a lot about research.
                            </Typography.Text>
                        </TimelineItemLong2>

                        <TimelineItemLong2>
                            <Typography.Text >
                                Built a data collection system consisting of a FMCW radar, a LiDAR, and a stereo camera based on ROS and collected a novel real-world perception dataset for autonomous driving.
                                Trained monocular and radar-based 3D object detectors to validate the effectiveness of the dataset.
                                <Anchor href="https://arxiv.org/abs/2311.10261" style={{ color: "darkblue", marginLeft: "2px" }}>CRUW3D (IEEE IV)</Anchor>
                            </Typography.Text>
                        </TimelineItemLong2>


                        <TimelineItemShort dot={<FileOutlined style={{ fontSize: '14px' }} />}>
                            <Typography.Text style={{ color: "purple" }}>
                                Master Degree at ECE, UW, 2023-06
                            </Typography.Text>
                        </TimelineItemShort>

                        <TimelineItemMedium dot={<HomeOutlined style={{ fontSize: '14px' }} />}>
                            <Typography.Text style={{ color: "purple" }}>
                                Started my PhD study at UW focusing on Computer Vision and Deep Learning, 2023-06
                            </Typography.Text>
                        </TimelineItemMedium>

                        <TimelineItemLong1>
                            <Typography.Text >
                                Created novel radar-based 3D object detector and object tracking system and achieved state-of-the-art performance on the competitive K-Radar benchmark.
                                <Link
                                    to={{
                                        pathname: `/project`,
                                        hash: PROJECTS.CENTER_RADAR_NET
                                    }}
                                    className="link"
                                    style={{ color: "darkblue", marginLeft: "2px" }}
                                >
                                    CenterRadarNet (IEEE ICIP)
                                </Link>
                            </Typography.Text>
                        </TimelineItemLong1>

                        <TimelineItemLong1>
                            <Typography.Text >
                                Worked on post-fusion 3D object detection and tracking system where radar and camera help each other to overcome the hard perception scenarios like snowy and foggy weather.
                                <Link
                                    to={{
                                        pathname: `/project`,
                                        hash: PROJECTS.CRAFTBOOSTER
                                    }}
                                    className="link"
                                    style={{ color: "darkblue", marginLeft: "2px" }}
                                >
                                    CRAFTBooster (IEEE IV)
                                </Link>
                            </Typography.Text>
                        </TimelineItemLong1>


                        <TimelineItemMedium dot={<LikeOutlined style={{ fontSize: '14px', color: "green" }} />}>
                            <Typography.Text >
                                Passed my qualification exam
                                <br />
                                Committee: Dr. Radha Poovendran, Dr. Linda Shapiro, Dr. Rania Hussein
                                , 2024-05
                            </Typography.Text>
                        </TimelineItemMedium>

                        <TimelineItemLong2>
                            <Typography.Text >
                                Co-led a research project to develope FMCW-radar-based human pose estimation system. Collected a dataset with synchronized radar, stereo cameras and LiDAR point clouds, and designed a novel human pose estimation model which surpases other radar-based methods.
                                <Link
                                    to={{
                                        pathname: `/project`,
                                        hash: PROJECTS.RTPOSE
                                    }}
                                    className="link"
                                    style={{ color: "darkblue", marginLeft: "2px" }}
                                >
                                    RTPOSE (ECCV)
                                </Link>
                            </Typography.Text>
                        </TimelineItemLong2>

                        <TimelineItemLong2>
                            <Typography.Text >
                                Worked on a research project to develop a novel teacher-student framewrok, MonoTAKD, to better distill knowledge from pre-trained LiDAR-based 3D object detectors to monocular 3D object detectors.  MonoTAKD achieves state-of-the-art performance on the competitive KITTI benchmark.
                                <Link
                                    to={{
                                        pathname: `/project`,
                                        hash: PROJECTS.MONOTAKD
                                    }}
                                    className="link"
                                    style={{ color: "darkblue", marginLeft: "2px" }}
                                >
                                    MonoTAKD (CVPR)
                                </Link>
                            </Typography.Text>
                        </TimelineItemLong2>

                        <TimelineItemLong1>
                            <Typography.Text >
                                Joined Microsoft as a research intern at the Phi-3 Vision team to work on multi-modal language models. I was fortunate to have <Anchor href="https://scholar.google.com/citations?user=M_fwZg0AAAAJ" style={{ color: "darkblue" }}>Qin Cai</Anchor>, <Anchor href="https://scholar.google.com/citations?user=jI5oDhYAAAAJ" style={{ color: "darkblue" }}>Yi-Ling Chen</Anchor> and <Anchor href="https://vibhav-vineet.github.io/" style={{ color: "darkblue" }}>Vibhav Vineet</Anchor> as my mentors.
                                <br />
                                2024-06~2024-12
                            </Typography.Text>
                        </TimelineItemLong1>


                        <TimelineItemLong2>
                            <Typography.Text >
                                I am currently studying and working on <Anchor href="https://youtu.be/uhLDHA9skFk?si=i6hXbsjzVfS57GWb" style={{ color: "darkblue" }}>Physical AI</Anchor>, with a focus on enhancing robots’ ability to perceive the world and take appropriate actions through improved multi-modal language-action models. I firmly believe that Physical AI is the future of both industry and the global economy, and I’m excited to be part of this evolving journey.
                                <br />
                                If you’re interested in this area, feel free to reach out to me via <Anchor href="mailto:andyhci@uw.edu" style={{ color: "darkblue" }}>email</Anchor>—let’s collaborate and push the boundaries of Physical AI together!
                                <br />
                                2025-03 ~ Present
                            </Typography.Text>
                        </TimelineItemLong2>
                    </Timeline>
                    {/* <Typography.Title level={5} style={{ color: "gray", textAlign: "center" }}>
                        .
                        <br />
                        .
                        <br />
                        .
                        <br />
                        To be continued...
                    </Typography.Title> */}
                </Paragraph>
                <Typography.Title level={4} style={{ color: "Purple", textAlign: "center" }}>
                    Go Dawgs!
                    <br />
                    <UWHuskyIcon />
                </Typography.Title>
            </ContentInner>
        </ContentOuter>
    </Container>
);

export default About;
