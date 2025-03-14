class Resource:
    def __init__(self, ri, ra, rp, rw, rm, rl, ru, rt, re=None):
        self.id = ri  # 자원 식별자
        self.activation_cost = ra  # 활성화 비용
        self.periodic_cost = rp  # 주기적 비용
        self.active_turns = rw  # 활성 턴 수
        self.maintenance_turns = rm  # 유지보수 턴 수
        self.lifespan = rl  # 총 수명
        self.power_units = ru  # 공급 가능한 건물 수
        self.effect_type = rt  # 특수 효과 유형
        self.effect_value = re  # 효과 값(퍼센트 또는 용량)

        # 자원의 상태 정보
        self.remaining_life = rl  # 남은 수명
        self.current_state = 'active'  # 현재 상태 (active/maintenance)
        self.state_counter = rw  # 현재 상태에서 남은 턴 수
        self.is_green = re > 0 if re is not None else None  # 녹색 자원 여부


class Accumulator:
    def __init__(self, capacity):
        self.stored = 0
        self.capacity = capacity


class GreenRevolutionGame:
    def __init__(self, initial_budget, resources, turns_data):
        self.initial_budget = initial_budget
        self.budget = initial_budget
        self.resources_definitions = resources
        self.turns_data = turns_data
        self.active_resources = []
        self.turn_number = 0
        self.accumulators = []
        self.score = 0
        self.purchases = [[] for _ in range(len(turns_data))]

    def buy_resource(self, resource_id):
        """자원을 구매하는 함수"""
        resource_def = next((r for r in self.resources_definitions if r.id == resource_id), None)
        if not resource_def or resource_def.activation_cost > self.budget:
            return False

        # 자원 복사본 생성
        resource = Resource(
            resource_def.id, resource_def.activation_cost, resource_def.periodic_cost,
            resource_def.active_turns, resource_def.maintenance_turns, resource_def.lifespan,
            resource_def.power_units, resource_def.effect_type, resource_def.effect_value
        )

        # C 타입(Maintenance Plan) 효과 적용
        if self.has_active_effect('C'):
            c_effects = [r for r in self.active_resources
                         if r.effect_type == 'C' and r.current_state == 'active']
            for c_effect in c_effects:
                if c_effect.is_green:  # 녹색 자원
                    resource.lifespan = resource.lifespan * (1 + c_effect.effect_value / 100)
                    resource.remaining_life = resource.lifespan
                else:  # 비녹색 자원
                    resource.lifespan = max(1, int(resource.lifespan * (1 - c_effect.effect_value / 100)))
                    resource.remaining_life = resource.lifespan

        # E 타입(Accumulator) 처리
        if resource.effect_type == 'E':
            self.accumulators.append(Accumulator(resource.effect_value))

        self.active_resources.append(resource)
        self.budget -= resource.activation_cost
        self.purchases[self.turn_number].append(resource_id)
        return True

    def update_resource_states(self):
        """모든 자원의 상태를 업데이트하는 함수"""
        resources_to_remove = []

        for resource in self.active_resources:
            # 수명 감소
            resource.remaining_life -= 1

            # 상태 카운터 감소
            resource.state_counter -= 1

            # 상태 변경 확인
            if resource.state_counter <= 0:
                if resource.current_state == 'active':
                    resource.current_state = 'maintenance'
                    resource.state_counter = resource.maintenance_turns
                else:  # 'maintenance'
                    resource.current_state = 'active'
                    resource.state_counter = resource.active_turns

            # 수명이 다한 자원 확인
            if resource.remaining_life <= 0:
                resources_to_remove.append(resource)

                # E 타입 자원(Accumulator)의 경우 저장된 전력을 다른 축전기로 이동
                if resource.effect_type == 'E':
                    accumulator = next((a for a in self.accumulators if a.capacity == resource.effect_value), None)
                    if accumulator:
                        self.accumulators.remove(accumulator)
                        # 남은 축전기가 있으면 저장된 전력 이동
                        if self.accumulators:
                            remaining_accumulators = self.accumulators[:]
                            stored = accumulator.stored
                            for acc in remaining_accumulators:
                                transfer = min(stored, acc.capacity - acc.stored)
                                acc.stored += transfer
                                stored -= transfer
                                if stored == 0:
                                    break

        # 수명이 다한 자원 제거
        for resource in resources_to_remove:
            self.active_resources.remove(resource)

    def calculate_powered_buildings(self):
        """이번 턴에 공급 가능한 건물 수 계산"""
        powered = 0

        # A 타입 효과(Smart Meter) 계산
        a_effect_multiplier = 1.0
        for resource in self.active_resources:
            if resource.effect_type == 'A' and resource.current_state == 'active':
                if resource.is_green:  # 녹색 자원
                    a_effect_multiplier += resource.effect_value / 100
                else:  # 비녹색 자원
                    a_effect_multiplier -= resource.effect_value / 100
        a_effect_multiplier = max(0, a_effect_multiplier)  # 음수 방지

        # 각 자원에서 공급 가능한 건물 수 계산
        for resource in self.active_resources:
            if resource.current_state == 'active' and resource.effect_type != 'E':
                # A 효과(Smart Meter) 적용
                resource_power = int(resource.power_units * a_effect_multiplier)
                powered += resource_power

        return powered

    def has_active_effect(self, effect_type):
        """특정 효과 타입이 활성화되어 있는지 확인"""
        return any(r.effect_type == effect_type and r.current_state == 'active' for r in self.active_resources)

    def get_active_effects(self, effect_type):
        """특정 효과 타입의 모든 활성화된 자원 반환"""
        return [r for r in self.active_resources if r.effect_type == effect_type and r.current_state == 'active']

    def calculate_turn_thresholds(self, base_min, base_max):
        """B 타입 효과(Distribution Facility)를 적용한 최소/최대 임계값 계산"""
        min_threshold = base_min
        max_threshold = base_max

        # B 효과(Distribution Facility) 적용
        for resource in self.active_resources:
            if resource.effect_type == 'B' and resource.current_state == 'active':
                if resource.is_green:  # 녹색 자원
                    min_threshold = min_threshold * (1 + resource.effect_value / 100)
                    max_threshold = max_threshold * (1 + resource.effect_value / 100)
                else:  # 비녹색 자원
                    min_threshold = min_threshold * (1 - resource.effect_value / 100)
                    max_threshold = max_threshold * (1 - resource.effect_value / 100)

        return max(0, int(min_threshold)), max(0, int(max_threshold))

    def calculate_profit_per_building(self, base_profit):
        """D 타입 효과(Renewable Plant)를 적용한 건물당 이익 계산"""
        profit = base_profit

        # D 효과(Renewable Plant) 적용
        for resource in self.active_resources:
            if resource.effect_type == 'D' and resource.current_state == 'active':
                if resource.is_green:  # 녹색 자원
                    profit = profit * (1 + resource.effect_value / 100)
                else:  # 비녹색 자원
                    profit = profit * (1 - resource.effect_value / 100)

        return max(0, int(profit))

    def use_accumulator(self, needed_buildings, powered_buildings):
        """축전기에서 필요한 만큼의 건물 공급량 확보"""
        if not self.accumulators:
            return 0

        # 필요한 건물 수 계산
        needed = max(0, needed_buildings - powered_buildings)
        if needed == 0:
            return 0

        # 축전기에서 사용 가능한 에너지 계산
        available = 0
        for acc in self.accumulators:
            available += acc.stored

        # 실제 사용할 에너지 결정
        used = min(needed, available)

        # 축전기에서 에너지 차감
        remaining = used
        for acc in self.accumulators:
            use_from_this = min(remaining, acc.stored)
            acc.stored -= use_from_this
            remaining -= use_from_this
            if remaining == 0:
                break

        return used

    def store_excess_in_accumulator(self, excess):
        """남은 에너지를 축전기에 저장"""
        if not self.has_active_effect('E') or excess <= 0:
            return

        # 활성화된 축전기에 에너지 저장
        remaining = excess
        for acc in self.accumulators:
            space = acc.capacity - acc.stored
            store_here = min(remaining, space)
            acc.stored += store_here
            remaining -= store_here
            if remaining == 0:
                break

    def calculate_maintenance_costs(self):
        """모든 활성 자원의 유지보수 비용 계산"""
        return sum(r.periodic_cost for r in self.active_resources)

    def execute_turn(self):
        """현재 턴을 실행하고 결과 반환"""
        if self.turn_number >= len(self.turns_data):
            return False

        turn_data = self.turns_data[self.turn_number]
        base_min_threshold, base_max_threshold, base_profit_per_building = turn_data

        # B 효과(Distribution Facility)를 적용한 임계값 계산
        min_threshold, max_threshold = self.calculate_turn_thresholds(base_min_threshold, base_max_threshold)

        # D 효과(Renewable Plant)를 적용한 이익 계산
        profit_per_building = self.calculate_profit_per_building(base_profit_per_building)

        # 공급 가능한 건물 수 계산
        powered_buildings = self.calculate_powered_buildings()

        # 축전기 사용
        extra_from_accumulator = 0
        if powered_buildings < min_threshold:
            extra_from_accumulator = self.use_accumulator(min_threshold, powered_buildings)

        total_buildings_powered = powered_buildings + extra_from_accumulator

        # 이익 계산
        turn_profit = 0
        if total_buildings_powered >= min_threshold:
            buildings_for_profit = min(total_buildings_powered, max_threshold)
            turn_profit = buildings_for_profit * profit_per_building

            # 초과 에너지를 축전기에 저장
            excess = max(0, powered_buildings - max_threshold)
            self.store_excess_in_accumulator(excess)

        # 유지보수 비용 계산
        maintenance_costs = self.calculate_maintenance_costs()

        # 예산 업데이트
        self.budget = self.budget + turn_profit - maintenance_costs

        # 점수 업데이트
        self.score += turn_profit

        # 자원 상태 업데이트
        self.update_resource_states()

        # 턴 증가
        self.turn_number += 1

        return True

    def simulate_game(self, purchase_strategy):
        """주어진 구매 전략에 따라 게임 시뮬레이션"""
        self.budget = self.initial_budget
        self.active_resources = []
        self.accumulators = []
        self.turn_number = 0
        self.score = 0
        self.purchases = [[] for _ in range(len(self.turns_data))]

        for turn, resources_to_buy in enumerate(purchase_strategy):
            self.turn_number = turn

            # 자원 구매
            total_activation_cost = sum(next(r.activation_cost for r in self.resources_definitions if r.id == res_id)
                                        for res_id in resources_to_buy)

            if total_activation_cost <= self.budget:
                for resource_id in resources_to_buy:
                    self.buy_resource(resource_id)

            # 턴 실행
            self.execute_turn()

        # 남은 턴 실행
        while self.turn_number < len(self.turns_data):
            self.execute_turn()

        return self.score, self.purchases

    def calculate_resource_value(self, resource, remaining_turns, future_demand):
        """
        자원의 가치를 계산하는 함수

        Args:
            resource: 평가할 자원
            remaining_turns: 게임에 남은 턴 수
            future_demand: 향후 예상 수요 (최소 임계값들의 평균)

        Returns:
            자원의 상대적 가치 점수
        """
        # 기본 효율: 전력 생산량 / 활성화 비용
        power_efficiency = resource.power_units / max(1, resource.activation_cost)

        # 자원 수명이 남은 턴 수보다 적으면 가치 감소
        lifespan_factor = min(1.0, resource.lifespan / max(1, remaining_turns))

        # 유지보수 비용 고려 (낮을수록 좋음)
        maintenance_penalty = 1.0 / (1.0 + resource.periodic_cost / 10.0)

        # 활성/유지보수 비율 (활성 턴이 많을수록 좋음)
        active_ratio = resource.active_turns / max(1, resource.active_turns + resource.maintenance_turns)

        # 특수 효과 타입에 따른 가치 계산
        effect_value = 1.0
        if resource.effect_type == 'A':  # Smart Meter
            if resource.is_green:  # 녹색 자원
                effect_value = 2.0 + (resource.effect_value / 100)
            else:  # 비녹색 자원
                effect_value = 0.5
        elif resource.effect_type == 'B':  # Distribution Facility
            if resource.is_green:  # 녹색 자원
                effect_value = 1.5 + (resource.effect_value / 200)
            else:  # 비녹색 자원
                effect_value = 0.7
        elif resource.effect_type == 'C':  # Maintenance Plan
            if resource.is_green:  # 녹색 자원
                effect_value = 1.8 + (resource.effect_value / 150)
            else:  # 비녹색 자원
                effect_value = 0.6
        elif resource.effect_type == 'D':  # Renewable Plant
            if resource.is_green:  # 녹색 자원
                effect_value = 2.2 + (resource.effect_value / 100)
            else:  # 비녹색 자원
                effect_value = 0.5
        elif resource.effect_type == 'E':  # Accumulator
            # 축전기는 향후 수요가 높을수록 중요해짐
            effect_value = 1.0 + (resource.effect_value / 10) * (future_demand / 20)

        # 전력 생산량이 미래 예상 수요보다 작으면 가치 증가
        demand_match = 1.0
        if resource.power_units > 0:  # 전력 생산 자원인 경우
            if resource.power_units < future_demand:
                # 수요에 일부 기여하는 경우 가치 증가
                demand_match = 1.5
            elif resource.power_units >= future_demand:
                # 수요를 충족할 수 있는 경우 매우 가치 있음
                demand_match = 2.0

        # 최종 가치 계산
        value = (
                power_efficiency *
                lifespan_factor *
                maintenance_penalty *
                active_ratio *
                effect_value *
                demand_match
        )

        return value

    def advanced_strategy(self):
        """
        향상된 자원 구매 전략 구현

        향후 수요를 예측하고, 자원의 다양한 특성을 고려하여 가치를 계산합니다.
        """
        # 전략 초기화
        strategy = [[] for _ in range(len(self.turns_data))]

        # 시뮬레이션을 위한 게임 객체 생성
        simulation = GreenRevolutionGame(self.initial_budget, self.resources_definitions, self.turns_data)

        # 향후 수요 예측 (모든 턴의 최소 임계값 평균)
        future_demand = sum(min_threshold for min_threshold, _, _ in self.turns_data) / len(self.turns_data)

        # 각 턴마다 전략 결정
        for turn in range(len(self.turns_data)):
            simulation.turn_number = turn
            current_budget = simulation.budget
            remaining_turns = len(self.turns_data) - turn

            # 현재 활성화된 자원의 정보를 바탕으로 필요 자원 분석
            current_power = simulation.calculate_powered_buildings()

            # 현재 턴의 수요(최소 임계값)
            current_demand, max_threshold, _ = simulation.turns_data[turn]

            # 파워가 부족한 경우 파워 생산에 우선순위
            power_priority = 1.0
            if current_power < current_demand:
                power_priority = 2.0  # 전력 부족 시 생산 자원 우선

            # 예산 관련 계수
            budget_threshold = 0.3  # 기본 예산 사용 비율

            # 후반부에 예산 사용 비율 증가
            if turn > len(self.turns_data) * 0.7:
                budget_threshold = 0.5

            # 전력이 심각하게 부족하면 예산 더 많이 사용
            if current_power < current_demand * 0.5:
                budget_threshold = min(0.7, budget_threshold + 0.2)

            # 구매 가능한 자원 정렬
            affordable_resources = [
                r for r in simulation.resources_definitions
                if r.activation_cost <= current_budget
            ]

            # 미래 수요에 맞춰 자원 가치 계산 및 정렬
            resources_by_value = sorted(
                affordable_resources,
                key=lambda r: (
                        self.calculate_resource_value(r, remaining_turns, future_demand) *
                        (power_priority if r.power_units > 0 else 1.0)
                ),
                reverse=True  # 높은 가치 순으로 정렬
            )

            # 구매 결정
            turn_purchases = []
            for resource in resources_by_value:
                if resource.activation_cost <= current_budget:
                    # 특수 효과 자원의 경우, 한 턴에 한 개만 구매
                    if resource.effect_type in ['A', 'B', 'C', 'D'] and any(
                            simulation.resources_definitions[res_id].effect_type == resource.effect_type
                            for res_id in turn_purchases
                    ):
                        continue

                    turn_purchases.append(resource.id)
                    current_budget -= resource.activation_cost

                    # 예산 제한 초과시 중단
                    if current_budget < simulation.budget * (1 - budget_threshold):
                        break

            strategy[turn] = turn_purchases

            # 턴 실행
            for res_id in turn_purchases:
                simulation.buy_resource(res_id)
            simulation.execute_turn()

        return strategy

    def genetic_algorithm_strategy(self, population_size=20, generations=10):
        """
        유전 알고리즘을 사용한 최적화 전략

        Args:
            population_size: 인구 크기
            generations: 세대 수

        Returns:
            최적화된 구매 전략
        """
        import random
        import copy

        # 한 턴에 구매할 수 있는 최대 자원 수
        max_resources_per_turn = 3

        # 초기 인구 생성
        population = []

        # 그리디 전략을 초기 인구에 포함
        greedy_strategy = self.greedy_strategy()
        population.append(greedy_strategy)

        # 나머지 인구는 랜덤 전략으로 생성
        for _ in range(population_size - 1):
            strategy = [[] for _ in range(len(self.turns_data))]

            for turn in range(len(self.turns_data)):
                # 이 턴에 살 자원의 수
                num_resources = random.randint(0, max_resources_per_turn)
                if num_resources > 0:
                    # 랜덤하게 자원 선택
                    available_resources = [r.id for r in self.resources_definitions]
                    strategy[turn] = random.sample(available_resources, min(num_resources, len(available_resources)))

            population.append(strategy)

        # 최고 점수와 전략 초기화
        best_score = 0
        best_strategy = None

        # 유전 알고리즘 실행
        for generation in range(generations):
            # 각 전략의 점수 평가
            scores = []
            for strategy in population:
                score, _ = self.simulate_game(strategy)
                scores.append(score)

                # 최고 점수 갱신
                if score > best_score:
                    best_score = score
                    best_strategy = copy.deepcopy(strategy)

            # 점수를 기준으로 전략 정렬
            sorted_population = [x for _, x in sorted(zip(scores, population), key=lambda pair: pair[0], reverse=True)]

            # 상위 30%만 선택
            elite_size = max(2, int(population_size * 0.3))
            elite = sorted_population[:elite_size]

            # 새로운 인구 생성
            new_population = list(elite)  # 엘리트는 그대로 유지

            # 교차 및 돌연변이로 나머지 채우기
            while len(new_population) < population_size:
                # 두 부모 선택
                parent1 = random.choice(elite)
                parent2 = random.choice(elite)

                # 교차
                child = [[] for _ in range(len(self.turns_data))]
                for turn in range(len(self.turns_data)):
                    if random.random() < 0.5:
                        child[turn] = copy.deepcopy(parent1[turn])
                    else:
                        child[turn] = copy.deepcopy(parent2[turn])

                # 돌연변이 (10% 확률)
                for turn in range(len(self.turns_data)):
                    if random.random() < 0.1:
                        # 이 턴에 살 자원의 수 변경
                        num_resources = random.randint(0, max_resources_per_turn)
                        if num_resources > 0:
                            # 랜덤하게 자원 선택
                            available_resources = [r.id for r in self.resources_definitions]
                            child[turn] = random.sample(available_resources,
                                                        min(num_resources, len(available_resources)))
                        else:
                            child[turn] = []

                new_population.append(child)

            # 인구 교체
            population = new_population

        # 최고 전략 반환
        return best_strategy

    def hybrid_strategy(self):
        """
        그리디 전략과 발전된 전략, 유전 알고리즘을 조합한 하이브리드 전략

        Returns:
            최종 구매 전략
        """
        # 그리디 전략 획득
        greedy_solution = self.greedy_strategy()
        greedy_score, _ = self.simulate_game(greedy_solution)

        # 발전된 전략 획득
        advanced_solution = self.advanced_strategy()
        advanced_score, _ = self.simulate_game(advanced_solution)

        # 유전 알고리즘 전략 획득 (턴 수가 많으면 세대 수 줄임)
        generations = 10
        if len(self.turns_data) > 200:
            generations = 5

        genetic_solution = self.genetic_algorithm_strategy(generations=generations)
        genetic_score, _ = self.simulate_game(genetic_solution)

        # 가장 높은 점수의 전략 선택
        best_strategy = greedy_solution
        best_score = greedy_score

        if advanced_score > best_score:
            best_strategy = advanced_solution
            best_score = advanced_score

        if genetic_score > best_score:
            best_strategy = genetic_solution

        return best_strategy

    def output_solution(self, purchases):
        """구매 결과를 출력 형식에 맞게 반환"""
        output_lines = []
        for turn, resources in enumerate(purchases):
            if resources:  # 구매한 자원이 있는 턴만 출력
                line = f"{turn} {len(resources)} " + " ".join(str(res_id) for res_id in resources)
                output_lines.append(line)
        return "\n".join(output_lines)

    def greedy_strategy(self):
        """간단한 그리디 전략 구현"""
        strategy = [[] for _ in range(len(self.turns_data))]
        simulation = GreenRevolutionGame(self.initial_budget, self.resources_definitions, self.turns_data)

        # 각 턴마다 수행
        for turn in range(len(self.turns_data)):
            simulation.turn_number = turn
            current_budget = simulation.budget

            # 가성비가 좋은 자원 순으로 정렬
            resources_by_value = sorted(
                [r for r in simulation.resources_definitions if r.activation_cost <= current_budget],
                key=lambda r: (r.power_units / max(1, r.activation_cost), -r.activation_cost),
                reverse=True
            )

            # 구매 가능한 만큼 자원 구매
            turn_purchases = []
            for resource in resources_by_value:
                if resource.activation_cost <= current_budget:
                    turn_purchases.append(resource.id)
                    current_budget -= resource.activation_cost

                    # 예산의 절반 이상 사용했으면 중지
                    if current_budget < simulation.budget / 2:
                        break

            strategy[turn] = turn_purchases

            # 턴 실행
            for res_id in turn_purchases:
                simulation.buy_resource(res_id)
            simulation.execute_turn()

        return strategy


# 입력 파싱 함수
def parse_input(input_str):
    lines = input_str.strip().split("\n")
    initial_budget, num_resources, num_turns = map(int, lines[0].split())

    resources = []
    for i in range(1, num_resources + 1):
        res_data = lines[i].split()
        ri = int(res_data[0])  # 자원 ID
        ra = int(res_data[1])  # 활성화 비용
        rp = int(res_data[2])  # 주기적 비용
        rw = int(res_data[3])  # 활성 턴 수
        rm = int(res_data[4])  # 유지보수 턴 수
        rl = int(res_data[5])  # 총 수명
        ru = int(res_data[6])  # 공급 가능한 건물 수
        rt = res_data[7]  # 특수 효과 유형

        # 효과 값이 있는 경우
        re = None
        if len(res_data) > 8:
            re = int(res_data[8])

        resources.append(Resource(ri, ra, rp, rw, rm, rl, ru, rt, re))

    turns_data = []
    for i in range(num_resources + 1, num_resources + num_turns + 1):
        tm, tx, tr = map(int, lines[i].split())
        turns_data.append((tm, tx, tr))

    return initial_budget, resources, turns_data


# 메인 실행 함수
def run_optimization():
    """
    모든 테스트 케이스에 대해 최적화 전략 실행 및 결과 출력
    """
    import os
    import time

    # 입력 파일 리스트
    input_files = [
        "0-demo.txt",
        "1-thunberg.txt",
        "2-attenborough.txt",
        "3-goodall.txt",
        "4-maathai.txt",
        "5-carson.txt",
        "6-earle.txt",
        "7-mckibben.txt",
        "8-shiva.txt"
    ]

    # 결과 저장
    results = {}

    # 각 입력 파일에 대해 최적화 실행
    for input_file in input_files:
        try:
            print(f"파일 처리 중: {input_file}")
            start_time = time.time()

            # 입력 파일 읽기
            input_path = f"./input/{input_file}"
            if not os.path.exists(input_path):
                print(f"파일이 존재하지 않음: {input_path}")
                continue

            with open(input_path, "r") as f:
                input_data = f.read()

            # 입력 파싱
            initial_budget, resources, turns_data = parse_input(input_data)

            # 게임 객체 생성
            game = GreenRevolutionGame(initial_budget, resources, turns_data)

            # 전략 선택 (케이스별 최적 전략 사용)
            if len(turns_data) <= 50:  # 작은 케이스는 하이브리드 전략
                strategy = game.hybrid_strategy()
            elif len(turns_data) <= 200:  # 중간 규모 케이스는 발전된 전략
                strategy = game.advanced_strategy()
            else:  # 큰 케이스는 그리디 전략 (시간 효율성)
                strategy = game.greedy_strategy()

            # 점수 계산
            score, purchases = game.simulate_game(strategy)

            # 결과 저장
            results[input_file] = score

            # 출력 저장
            output_path = f"./output/{input_file}"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
                f.write(game.output_solution(purchases))

            # 실행 시간 및 점수 출력
            elapsed_time = time.time() - start_time
            print(f"- 점수: {score}")
            print(f"- 소요 시간: {elapsed_time:.2f}초")

        except Exception as e:
            print(f"오류 발생 ({input_file}): {str(e)}")

    # 최종 결과 출력
    print("\n최종 결과:")
    for input_file, score in results.items():
        print(f"{input_file}: {score}")


if __name__ == "__main__":
    run_optimization()