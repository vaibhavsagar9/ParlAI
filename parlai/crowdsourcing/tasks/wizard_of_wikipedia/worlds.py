#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.core.agents import create_agent
import copy
import os
from parlai.mturk.core.worlds import MTurkOnboardWorld, MTurkTaskWorld
from parlai.core.worlds import validate
from joblib import Parallel, delayed


def setup_retriever(opt):
    print("[ Setting up Retriever ]")
    ret_opt = copy.deepcopy(opt)
    ret_opt["model_file"] = "models:wikipedia_full/tfidf_retriever/model"
    ret_opt["retriever_num_retrieved"] = opt.get("num_passages_retrieved", 2)
    ret_opt["retriever_mode"] = "keys"
    ret_opt["override"] = {"remove_title": False}
    ret_opt["datapath"] = os.path.join("/scratch/komeili", "datapath", "wikiwiz")
    ir_agent = create_agent(ret_opt)
    print("-" * 40 + "\n[ Retriever setup successfully]")
    return ir_agent


class MTurkMultiAgentDialogOnboardWorld(MTurkOnboardWorld):
    def __init__(self, opt, mturk_agent):
        super().__init__(opt, mturk_agent)
        self.opt = opt

    def parley(self):
        self.mturk_agent.agent_id = "Onboarding Agent"
        self.mturk_agent.observe({"id": "System", "text": "Welcome onboard!"})
        x = self.mturk_agent.act(timeout=self.opt["turn_timeout"])
        self.mturk_agent.observe(
            {
                "id": "System",
                "text": "Thank you for your input! Please wait while "
                "we match you with another worker...",
                "episode_done": True,
            }
        )
        self.episodeDone = True


class MTurkMultiAgentDialogWorld(MTurkTaskWorld):
    """
    Basic world where each agent gets a turn in a round-robin fashion, receiving as
    input the actions of all other agents since that agent last acted.
    """

    def __init__(self, opt, agents=None, shared=None):
        # Add passed in agents directly.
        self.agents = agents
        self.acts = [None] * len(agents)
        self.episodeDone = False
        self.max_turns = opt.get("max_turns", 2)
        self.current_turns = 0
        self.send_task_data = opt.get("send_task_data", False)
        self.opt = opt
        for idx, agent in enumerate(self.agents):
            agent.agent_id = f"Chat Agent {idx + 1}"

    def parley(self):
        """
        For each agent, get an observation of the last action each of the other agents
        took.
        Then take an action yourself.
        """
        acts = self.acts
        self.current_turns += 1
        for index, agent in enumerate(self.agents):
            print(f'--- turn:{self.current_turns} id:{index} agent{agent}')
            try:
                acts[index] = agent.act(timeout=self.opt["turn_timeout"])
                if self.send_task_data:
                    acts[index].force_set(
                        "task_data",
                        {
                            "last_acting_agent": agent.agent_id,
                            "current_dialogue_turn": self.current_turns,
                            "utterance_count": self.current_turns + index,
                        },
                    )
            except TypeError:
                acts[index] = agent.act()  # not MTurkAgent

            for k, v in acts[index].items():
                print(f'{k} {type(v)}')

            if "candidate_ids" in acts[index]:
                print("nd array to list")
                acts[index]["candidate_ids"] = acts[index]["candidate_ids"].tolist()
                acts[index]["candidate_scores"] = acts[index][
                    "candidate_scores"
                ].tolist()

            if "episode_done" in acts[index] and acts[index]["episode_done"]:
                self.episodeDone = True
            for other_agent in self.agents:
                if other_agent != agent:
                    other_agent.observe(validate(acts[index]))

    def prep_save_data(self, agent):
        """Process and return any additional data from this world you may want to store"""
        return {"example_key": "example_value"}

    def episode_done(self):
        return self.episodeDone

    def shutdown(self):
        """
        Shutdown all mturk agents in parallel, otherwise if one mturk agent is
        disconnected then it could prevent other mturk agents from completing.
        """
        global shutdown_agent

        def shutdown_agent(agent):
            try:
                agent.shutdown(timeout=None)
            except Exception:
                agent.shutdown()  # not MTurkAgent

        Parallel(n_jobs=len(self.agents), backend="threading")(
            delayed(shutdown_agent)(agent) for agent in self.agents
        )


def make_onboarding_world(opt, agent):
    return MTurkMultiAgentDialogOnboardWorld(opt, agent)


def validate_onboarding(data):
    """Check the contents of the data to ensure they are valid"""
    print(f"Validating onboarding data {data}")
    return True


def make_world(opt, agents):
    print(opt)
    ir_agent = setup_retriever(opt)
    agents.append(ir_agent)
    return MTurkMultiAgentDialogWorld(opt, agents)


def get_world_params():
    return {"agent_count": 2}
