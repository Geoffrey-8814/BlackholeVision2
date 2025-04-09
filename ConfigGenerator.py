import json
import ntcore
import numpy as np
import convertor
import torch
import torch.multiprocessing as mp
import time

import Process

class configGenerator:
    def __init__(self):
        with open("config.json", "r") as config_file:
            config_data = json.loads(config_file.read())
            
        self.device_id = config_data['device_id']
        self.server_ip = config_data['server_ip']
        
        ntcore.NetworkTableInstance.getDefault().setServer(self.server_ip)
        ntcore.NetworkTableInstance.getDefault().startClient4(self.device_id)
        # ntcore.NetworkTableInstance.getDefault().setServerTeam(8814)
        
        self.nt_table = ntcore.NetworkTableInstance.getDefault().getTable(
            self.device_id
        )
        
        self.config: dict = {}
        
        self.camerasName_sub = self.nt_table.getStringArrayTopic("camerasName").subscribe([])
        self.camerasName = []
        
        self.tagLayout_sub = self.nt_table.getStringTopic("tagLayout").subscribe("")
        
        self.tagSize_sub = self.nt_table.getDoubleTopic("tagSize").subscribe(0)
        
        self.cameraMatrix_subs: dict = {}
        self.distortionCoeffs_subs: dict = {}
        self.cameraPose_subs: dict = {}
        self.resolution_subs: dict = {}
        self.exposure_subs: dict = {}
        self.gain_subs: dict = {}
        self.maxFPS_subs: dict = {}
        
        self.enableTag_subs: dict = {}
        self.enableObj_subs: dict = {}
        
        
        self.waitConfig()
        self.setup_subs()
        
        time.sleep(1)#wait for setup(ntcore subscriber)
        
        self.getStaticConfig()
        
    def waitConfig(self):
        self.camerasName = []
        while True:
            if ntcore.NetworkTableInstance.getDefault().isConnected():
                break
            print('wait for connection')
            time.sleep(1)
        while True:
            self.camerasName = self.camerasName_sub.get()
            if self.camerasName != []:
                break
            print('wait for config')
            time.sleep(1)
        
    def setup_subs(self):
        for name in self.camerasName:
            # cameraTable = ntcore.NetworkTableInstance.getDefault().getTable(
            #     "/" + self.device_id + "/" + name
            # )
            # cameraTable.getDoubleArrayTopic("test2").publish().set([0,0,0])
            # print(cameraTable.getDoubleArrayTopic("test2").subscribe([]).get())
            self.cameraMatrix_subs[name] = self.nt_table.getDoubleArrayTopic(name + "/cameraMatrix").subscribe([])
            self.distortionCoeffs_subs[name] = self.nt_table.getDoubleArrayTopic(name + "/distortionCoeffs").subscribe([])
            self.cameraPose_subs[name] = self.nt_table.getDoubleArrayTopic(name + "/cameraPose").subscribe([])
            self.resolution_subs[name] = self.nt_table.getIntegerArrayTopic(name + "/resolution").subscribe([])
            self.exposure_subs[name] = self.nt_table.getDoubleTopic(name + "/exposure").subscribe(0)
            self.gain_subs[name] = self.nt_table.getDoubleTopic(name + "/gain").subscribe(0)
            self.maxFPS_subs[name] = self.nt_table.getDoubleTopic(name + "/maxFPS").subscribe(0)
            
            self.enableTag_subs[name] = self.nt_table.getBooleanTopic(name + "/enableTag").subscribe(False)
            self.enableObj_subs[name] = self.nt_table.getBooleanTopic(name + "/enableObj").subscribe(False)
            
    def getStaticConfig(self):
        self.config['tagLayout'] = json.loads(self.tagLayout_sub.get())["tags"]
        self.config['tagSize'] = self.tagSize_sub.get()
        self.config["camerasName"] = self.camerasName
        for name in self.camerasName:
            self.config[name] = {}
            self.config[name]["cameraMatrix"] = np.array(self.cameraMatrix_subs[name].get()).reshape(3,3)
            self.config[name]["distortionCoeffs"] = np.array(self.distortionCoeffs_subs[name].get())
            self.config[name]["resolution"] = self.resolution_subs[name].get()
            
            self.config[name]["enableTag"] = self.enableTag_subs[name].get()
            self.config[name]["enableObj"] = self.enableObj_subs[name].get()
        
    def getDynamicConfig(self):
        for name in self.camerasName:
            self.config[name]["cameraPose"] = convertor.robotPoseToTensor(convertor.listToRobotPose(self.cameraPose_subs[name].get()))
            exposure = self.exposure_subs[name].get()
            gain = self.gain_subs[name].get()
            maxFPS = self.maxFPS_subs[name].get()
            self.config[name]["cameraConfig"] = torch.tensor([exposure, gain, maxFPS])
    
    def updateDynamicConfig(self, camerasConfigTensors, apriltagConfigTensors):
        self.getDynamicConfig()
        for name in self.camerasName:
            camerasConfigTensors[name]['config'].copy_(self.config[name]["cameraConfig"])
            apriltagConfigTensors[name]['config'].copy_(self.config[name]["cameraPose"])
    
    def getConfig(self):
        return self.config
    
    def getPosePublishers(self):
        self.posePublishers: dict = {}
        for name in self.camerasName:
            self.posePublishers[name] = {}
            self.posePublishers[name]['multiTagPose'] = self.nt_table.getDoubleArrayTopic(name + '/multiTagPose').publish(
                ntcore.PubSubOptions(periodic=0, sendAll=True, keepDuplicates=True))
            self.posePublishers[name]['multiTagError'] = self.nt_table.getDoubleTopic(name + '/multiTagError').publish(
                ntcore.PubSubOptions(periodic=0, sendAll=True, keepDuplicates=True))
            self.posePublishers[name]['latency']= self.nt_table.getDoubleTopic(name + '/latency').publish(
                ntcore.PubSubOptions(periodic=0, sendAll=True, keepDuplicates=True))
        return self.posePublishers
    
    def getObjPosePublishers(self):
        self.posePublishers: dict = {}
        for name in self.camerasName:
            self.posePublishers[name] = {}
            self.posePublishers[name]['coralPoses'] = self.nt_table.getDoubleArrayTopic(name + '/coralPoses').publish(
                ntcore.PubSubOptions(periodic=0, sendAll=True, keepDuplicates=True))
            self.posePublishers[name]['coralErrors'] = self.nt_table.getDoubleArrayTopic(name + '/coralErrors').publish(
                ntcore.PubSubOptions(periodic=0, sendAll=True, keepDuplicates=True))
            self.posePublishers[name]['algaePoses'] = self.nt_table.getDoubleArrayTopic(name + '/algaePoses').publish(
                ntcore.PubSubOptions(periodic=0, sendAll=True, keepDuplicates=True))
            self.posePublishers[name]['latency']= self.nt_table.getDoubleTopic(name + '/latency').publish(
                ntcore.PubSubOptions(periodic=0, sendAll=True, keepDuplicates=True))
        return self.posePublishers
    
    def getSharedTensorsAndEvents(self):
        camerasConfigTensors = {}
        captureTensors = {}
        captureEvents = {}
        apriltagConfigTensors = {}
        poseTensors = {}
        poseEvents = {}
        
        objPoseTensors = {}
        objPoseEvents = {}
        
        
        for name in self.camerasName:
            resolution = self.config[name]['resolution']
            H, W = resolution[0], resolution[1]
            camerasConfigTensors[name] = Process.getSharedTensors({
                'config': (3),
            })
            captureTensors[name] = Process.getSharedTensors({
                'frame': (H, W, 3),
                'metaData': (1)
            })
            captureEvents[name] = {
                'apriltag': mp.Event(),
                'ML': mp.Event(),
            }
            apriltagConfigTensors[name] = Process.getSharedTensors({
                'config': (6),
            })
            
            poseTensors[name] =  Process.getSharedTensors({
                'multiTagPose': (6),
                'multiTagError': (1),
                'latency': (1),
            })
            poseEvents[name] = {
                'publish': mp.Event(),
            }
            
            objPoseTensors[name] =  Process.getSharedTensors({
                'coralPoses': (10,4), # [x, y, Theta 1, Theta 2] * 10
                'coralErrors': (10),
                'algaePoses': (20), # [x, y] * 10
                'latency': (1),
            })
            
            objPoseEvents[name] = {
                'publish': mp.Event(),
            }
            
        return camerasConfigTensors, captureTensors, captureEvents, apriltagConfigTensors, poseTensors, poseEvents, objPoseTensors, objPoseEvents